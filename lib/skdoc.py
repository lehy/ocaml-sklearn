import io
import os
import abc
import inspect
import pkgutil
import pathlib
import logging
import textwrap
import distutils
import importlib
import functools
import collections
import regex as re

log = logging.getLogger('skdoc')


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
    if not s:
        s = 'T'
    ops = {
        '-': 'Minus',
        '+': 'Plus',
        ' ': 'Space',
        '*': 'Mult',
        '/': 'Div',
        '<': 'Lt',
        '>': 'Gt',
        '==': 'Eq',
        '<=': 'Lte',
        '>=': 'Gte',
        '!=': 'Neq',

        # Note that these could probably more explicit.
        'r': 'R',
        'w': 'W',
        'r+': 'R_plus',
        'w+': 'W_plus',
        'rw': 'RW',
        'a': 'A',
        'A+': 'A_plus'
    }
    if s in ops:
        return ops[s]
    s = re.sub(r'[^a-zA-Z0-9_]+', '_', s)
    s = re.sub(r'^([^a-zA-Z])', r'T\1', s)
    return ucfirst(s)


def mlid(s):
    if s is None:
        return None
    if s == '_':
        return '_dummy'
    # DESCR -> descr
    if s == s.upper():
        return s.lower()
    s = lcfirst(s)
    if s in ml_keywords:
        return s + '_'
    return s


def make_module_name(x):
    if not re.match(r'^[a-zA-Z]', x):
        return 'M' + x
    else:
        return ucfirst(x)


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


class TypeContext:
    def __init__(self, **kwargs):
        self.bindings = kwargs

    def add(self, **kwargs):
        ret = type(self)(**self.bindings)
        ret.bindings.update(kwargs)
        return ret

    def __getattr__(self, name):
        return self.bindings[name]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = "--------- context:\n"
        for k, v in self.bindings.items():
            ret += f"{k}:\n"
            ret += indent(maybe_to_string(v)) + "\n"
        return ret


class Registry:
    def __init__(self):
        self.types = collections.defaultdict(list)
        self.module_path = {}
        self.bases = collections.defaultdict(set)
        self.generated_files = []
        self.generated_doc_files = []
        self.elements = collections.defaultdict(list)
        
    def add(self, name, element):
        self.elements[element].append(name)
        
    def add_generated_file(self, f):
        f = pathlib.Path(f)
        assert f not in self.generated_files, f"source file already generated: {f}"
        self.generated_files.append(f)

    def add_generated_doc_file(self, f):
        f = pathlib.Path(f)
        assert f not in self.generated_doc_files, f"doc file already generated: {f}"
        self.generated_doc_files.append(f)

    def add_class(self, klass, module_path):
        if self.module_path.get(klass, module_path) != module_path:
            # existing = self.module_path[klass]
            # if 'Base' not in existing:
            #     print(
            #         f"WW {klass.__name__} is already registered with path {existing}, ignoring path {module_path}"
            #     )
            pass
        else:
            self.module_path[klass] = module_path
        for ancestor in interesting_ancestors(klass):
            self.types[ancestor].append(klass)
            self.bases[klass].add(ancestor)

    def write(self, build_dir):
        pass

    def report_generated(self):
        multiple = []
        for element, names in self.elements.items():
            if len(names) > 1:
                multiple.append((len(names), element, names))
        multiple.sort(key=lambda x: x[0])
        for num, element, names in multiple:
            ename = getattr(element, '__name__', maybe_to_string(element))
            log.warning(f"skdoc.py:0:{type(element).__name__} {ename}: was emitted under {num} names:")
            for name in names:
                log.warning(f"- {name.full_python_name()} / {name.full_ml_name()}")

        if self.generated_files:
            log.info("generated source files:")
            for f in sorted(self.generated_files):
                print(f)
        if self.generated_doc_files:
            log.info("generated doc files:")
            for f in sorted(self.generated_doc_files):
                print(f)


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
        # print(f"line: '{line}' indent: {indent} previous: {previous_indent}")

        if not line or indent < previous_indent:
            # log.debug("-> end paragraph")
            elements.append(Paragraph("\n".join(element)))
            element = []
            previous_indent = None
            if line:
                element.append(line)
        elif re.match(r'^-+$', line):
            # print(f'found underline, element: {element}')
            if not element:
                log.warning(f"found underline with nothing above:\n{doc}")
                continue
            block, title = element[:-1], element[-1]
            if block:
                elements.append(Paragraph("\n".join(block)))
            elements.append(Section_title(title))
            element = []
            previous_indent = None
        else:
            # log.debug("append line to existing element")
            element.append(line)
            previous_indent = indent
    if element:
        elements.append(Paragraph("\n".join(element)))

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
    text = re.sub(r'(,\s*)?\(\S+\s+by\s+default\)', '', text)
    text = re.sub(r'\s*\(default\)', '', text)
    text = re.sub(r'\s*\(if [^()]+\)', '', text)
    text = re.sub(r'[Dd]efaults\s+to\s+\S+\.?', '', text)
    text = re.sub(r'[Dd]efault\s+is\s+\S+\.?', '', text)
    text = re.sub(r'\(?\s*[Dd]efault\s*[:=]?\s*.+\)?$', '', text)
    text = re.sub(r'\s*optional', '', text)
    text = re.sub(r'(,\s*)?\S+\s+by\s+default', '', text)
    return text


class ReParser:
    """Parsing things with regexps, like shapes and enums. This is
       probably the cleanest I can do without turning to a proper
       parser generator.

    """
    def __init__(self):
        nob = r'(?:[^()[\]]*)'

        onep = rf'(?:\({nob}\))'
        oneb = rf'(?:\[{nob}\])'
        onec = rf'(?:\{{{nob}\}})'
        one = rf'(?:{oneb}|{onep}|{onec})'

        onepn = r'(?:\( (?:[nm]\w*,\s*)+ (?:[nm]\w*)? \))'
        onebn = r'(?:\[ (?:[nm]\w*,\s*)+ (?:[nm]\w*)? \])'
        onecn = r'(?:\{ (?:[nm]\w*,\s*)+ (?:[nm]\w*)? \})'
        onen = rf'(?:{onepn}|{onebn}|{onecn})'

        twop = rf'(?:\((?:{nob}{one})*{nob}?\))'
        twob = rf'(?:\[(?:{nob}{one})*{nob}?\])'
        twoc = rf'(?:\{{(?:{nob}{one})*{nob}?\}})'
        two = rf'(?:{twob}|{twop}|{twoc})'

        threep = rf'(?:\((?:{nob}{two})*{nob}?\))'
        threeb = rf'(?:\[(?:{nob}{two})*{nob}?\])'
        threec = rf'(?:\{{(?:{nob}{two})*{nob}?\}})'
        three = rf'(?:{threeb}|{threep}|{threec})'
        self.three = three

        ifelse = rf'(if \s+ \S+ \s* =+ \s* \S+ \s+ else \s* {three})'
        ort = rf'(,? \s* or \s* ({three}|smaller) (\s* if \s+ \S+ \s* =+ \s* \S+)?)'

        # Things inside parentheses, with several spaces but no comma (=>
        # not a tuple). There can be paren groups with commas inside them, though.
        p_no_tuple = rf'(\( ([^(),]* {three})* [^(),]* \s [^\s(),]+ \s ([^(),]* {three})* [^(),]* \))'

        res = [
            rf'(((of|with) \s+)? shape) (\s* ([=:]\s* | of \s*)?)? {three} (\s* ({ifelse}|{ort}))?',
            rf'(of \s*)? {onen} (?:\s* if \s+ \S+ \s* =+ \s* \S+)?',
            r'(of|with) \s+ length (\s+ | \s*[=:]\s*) \S+',
            r'\S+-dimension(al)?',
            p_no_tuple,
        ]
        shape = '|'.join([rf'({x})' for x in res])
        shape = rf'(,\s*)?(?:{shape})'

        self.shape = self._comp(shape)

        enum_elt = rf'(?P<elt>(?:(?:(?! (?:, \s* | \s) or \s+) [^()[\],])* {three})* (?:(?! (?:, \s* | \s) or \s+)[^()[\],])*)'  # noqa
        enum_comma = rf'(?:(?:{enum_elt},)+ {enum_elt})'
        enum_semi = rf'(?:(?:{enum_elt};)+ {enum_elt})'
        enum_pipe = rf'(?:(?:{enum_elt}\|)+ {enum_elt})'
        enum_or = rf'(?:(?:{enum_elt} (?:,\s* | \s) or \s)+ {enum_elt})'
        enum_comma_or = rf'(?:{enum_comma} (?:,\s* | \s) or \s {enum_elt})'

        string = r'''(?P<elt>"[^"]+" | '[^']+' | None) (?: \s* \( \s* default \s* \) \s*)?'''
        strings_comma = rf'(?:(?:{string} \s* , \s*)* {string})'
        strings_semi = rf'(?:(?:{string} \s* ; \s*)* {string})'
        strings_pipe = rf'(?:(?:{string} \s* \| \s*)* {string})'
        strings = rf'(?:\s* (?:{strings_comma} | {strings_semi} | {strings_pipe}) \s*)'
        strings_p = rf'(?:\( \s* {strings} \s* \))'
        strings_b = rf'(?:\[ \s* {strings} \s* \])'
        strings_c = rf'(?:\{{ \s* {strings} \s* \}})'
        string_enum = rf'(?:(?:str(ing)? (?:\s+ in | \s* [,:])? \s*)? (?:{strings_p} | {strings_b} | {strings_c} | {strings_pipe}))'  # noqa

        enum_c = rf'(?: \{{ (?:{enum_comma} | {enum_semi} | {enum_pipe}) \}})'

        enum = rf'^\s* (?:(?:string, \s*)? either \s+)? (?:{string_enum} | {enum_c} | {enum_comma} | {enum_pipe} | {enum_or} | {enum_comma_or}) \s*$'  # noqa e501
        self.string = self._comp(string)
        self.strings = self._comp(strings)
        self.strings_c = self._comp(strings_c)
        self.string_enum = self._comp(string_enum)
        self.enum_elt = self._comp(enum_elt)
        self.enum_comma = self._comp(enum_comma)
        self.enum_or = self._comp(enum_or)
        self.enum_comma_or = self._comp(enum_comma_or)
        self.enum = self._comp(enum)

        doc_sig = rf'^\s*(?:\w \s* (?:, \s* \w \s*)* = \s*)? ([\w\.]+)({three})'
        self.doc_sig = self._comp(doc_sig)

    def _comp(self, patt):
        return re.compile(patt, flags=re.VERBOSE | re.IGNORECASE)


re_parser = ReParser()


def remove_shape(text):
    text = re.sub(re_parser.shape, '', text)

    return text


def test_remove_shape():
    tests = [
        'shape [n,m]', 'of shape (n_samples,) or (n_samples, n_outputs)',
        'shape (n_samples, n_features)',
        '(len() equal to cv or 1 if cv == "prefit")', 'of shape [n_samples]',
        'of shape (n_class,)', 'shape (n_bins,) or smaller',
        'shape = [1, n_features] if n_classes == 2 else [n_classes, n_features]',
        'shape (n_samples, n_features), or             (n_samples, n_samples)',
        'shape of (n_samples, n_features)', 'of shape of (n_targets,)',
        'shape of (n_class * (n_class-1) / 2,)', 'of (N,)', 'of (n_features,)',
        """shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'""",
        '(n_samples, n_features)',
        "[n_samples_a, n_samples_a] if metric == 'precomputed'"
    ]

    success = True
    for test in tests:
        result = remove_shape(test)
        if result:
            log.error(f"test fails: '{test}' -> '{result}' (expected '')")
            success = False

    tests_no = [
        "string in ['raw_values', 'uniform_average',                 'variance_weighted']",
        'list of (int,) or list of (int, int)'
    ]
    for test in tests_no:
        result = remove_shape(test)
        if result != test:
            log.error(
                f"test fails: '{test}' -> '{result}' (expected '{test}')")
            success = False

    return success


def is_string(x):
    if not x:
        return False
    return (x[0] == x[-1] and x[0] in ("'", '"')) or x in [
        "eigen'"
    ]  # hack for doc bug of RidgeCV


def is_int(x):
    return re.match(r'^\d+$', x)


def parse_enum(t):
    m = re.match(re_parser.enum, t)
    if m is None:
        return None
    ret = m.captures("elt")
    ret = [x.strip() for x in ret]
    ret = [x for x in ret if x]
    return ret


def test_parse_enum():
    tests = [
        ('list of (int,) or list of (int, int)',
         ['list of (int,)', 'list of (int, int)']),
        ('list of (int,) or list of (int, int) or list of str',
         ['list of (int,)', 'list of (int, int)', 'list of str']),
        ('list of (int,), list of (int, int)',
         ['list of (int,)', 'list of (int, int)']),
        ('list of (int,), list of (int, int) or list of ndarray',
         ['list of (int,)', 'list of (int, int)', 'list of ndarray']),
        ("string in {'a', 'b'}", ["'a'", "'b'"]),
        ("str in {'a', 'b'}", ["'a'", "'b'"]),
        ("str, {'a', 'b'}", ["'a'", "'b'"]),
        ("string, {'a', 'b'}", ["'a'", "'b'"]),
        ("string in ['a', 'b']", ["'a'", "'b'"]),
        ("str in ['a', 'b']", ["'a'", "'b'"]),
        ("str, ['a', 'b']", ["'a'", "'b'"]),
        ("string, ['a', 'b']", ["'a'", "'b'"]),
        ("string in ('a', 'b')", ["'a'", "'b'"]),
        ("str in ('a', 'b')", ["'a'", "'b'"]),
        ("str, ('a', 'b')", ["'a'", "'b'"]),
        ("string, ('a', 'b')", ["'a'", "'b'"]),
        ("string : {'a', 'b'}", ["'a'", "'b'"]),
        ("str : {'a', 'b'}", ["'a'", "'b'"]),
        ("str: {'a', 'b'}", ["'a'", "'b'"]),
        ("{array-like, sparse matrix}", ['array-like', 'sparse matrix']),
        ("""string, [None, 'binary', 'micro', 'macro', 'samples',                        'weighted']""",
         ['None', "'binary'", "'micro'", "'macro'", "'samples'",
          "'weighted'"]),
        ("""string, [None, 'binary' (default), 'micro', 'macro', 'samples',                        'weighted']""",
         ['None', "'binary'", "'micro'", "'macro'", "'samples'",
          "'weighted'"]),
        ("{'raw_values', 'uniform_average'} or array-like of shape",
         ["{'raw_values', 'uniform_average'}", "array-like of shape"]),
        ('array-like  or BallTree', ['array-like', 'BallTree']),
        ('array-like,  BallTree', ['array-like', 'BallTree']),
        ('array-like,  BallTree or Tata', ['array-like', 'BallTree', 'Tata']),
        ('array-like,  BallTree,  or Tata', ['array-like', 'BallTree',
                                             'Tata']),
        ('array-like or  BallTree,  or Tata',
         ['array-like', 'BallTree', 'Tata']),
        ("'linear' | 'poly' | 'rbf' | 'sigmoid' | 'cosine' | 'precomputed'", [
            "'linear'", "'poly'", "'rbf'", "'sigmoid'", "'cosine'",
            "'precomputed'"
        ]), ("string, either '-', '+', or ' '", ["'-'", "'+'", "' '"])
    ]

    success = True
    for test, expected in tests:
        elts = parse_enum(test)
        if elts != expected:
            log.error(
                f"error in enum test: '{test}' -> {elts} (expected {expected})"
            )
            success = False

    return success


def partition(li, pred):
    sat = []
    unsat = []
    for x in li:
        if pred(x):
            sat.append(x)
        else:
            unsat.append(x)
    return sat, unsat


# this does not cover all cases it seems, where different instances of
# the same class with the same params are considered different?
# using elt.__class__ is appealing but would break StringValue for instance.
def remove_duplicates(elts):
    return list(set(elts))


# got = set()
# ret = []
# for elt in elts:
#     if elt not in got:
#         ret.append(elt)
#         got.add(elt)
# return ret


def simplify_arr(enum):
    arr, not_arr = partition(enum.elements,
                             lambda x: isinstance(x, (Arr, Ndarray)))
    sparse, not_arr_not_sparse = partition(
        not_arr, lambda x: isinstance(x, (SparseMatrix, CsrMatrix, CscMatrix)))
    if arr and sparse:
        return type(enum)([Arr()] + not_arr_not_sparse)
    else:
        return enum


def simplify_arr_or_float(enum):
    if len(enum.elements) != 2:
        return enum

    e0, e1 = enum.elements
    if ((isinstance(e0, Float) and isinstance(e1, Arr))
            or (isinstance(e1, Float) and isinstance(e0, Arr))):
        return type(enum)([Arr()])

    return enum


def simplify_enum(enum):
    # log.debug(f"simplifying enum %s", enum)

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
        bool, not_bool = partition(not_false_true,
                                   lambda x: isinstance(x, Bool))
        enum = Enum([Bool()] + not_bool)

    none, not_none = partition(enum.elements,
                               lambda x: isinstance(x, NoneValue))
    assert len(none) <= 1, enum
    if none:
        if not_none:
            inside = simplify_enum(type(enum)(not_none))
            return Optional(inside)
        else:
            return none
    # enum = EnumOption(not_none)
    # enum = type(enum)(not_none + [NoneValue()])

    # log.debug("enum 2: %s", enum)

    # Enum(String, StringValue, StringValue) should just be
    # Enum(StringValue, StringValue) since that is (probably) a
    # misparsing of "str, 'l1' or 'l2'"
    is_string_value, is_not_string_value = partition(
        enum.elements, lambda x: isinstance(x, StringValue))
    if is_string_value and len(is_not_string_value) == 1 and isinstance(
            is_not_string_value[0], String):
        enum = type(enum)(is_string_value)

    # Int|TupleOfInts -> TupleOfInts
    if len(enum.elements) == 2:
        a, b = enum.elements
        if isinstance(a, Int) and isinstance(b, TupleOfInts):
            return b
        if isinstance(b, Int) and isinstance(a, TupleOfInts):
            return a

    # log.debug("enum 3: %s", enum)

    enum = type(enum)(remove_duplicates(enum.elements))

    # Arr | SparseMatrix == Arr
    enum = simplify_arr(enum)

    # An Arr.t is able to represent a single Numpy scalar also.
    enum = simplify_arr_or_float(enum)

    # log.debug("enum 4: %s", enum)

    # There is no point having more than one Py.Object tag in an enum.
    is_obj, is_not_obj = partition(
        enum.elements, lambda x: isinstance(x, (PyObject, UnknownType)))
    if len(is_obj) > 1:
        enum = type(enum)(is_not_obj + [PyObject()])  # XXX

    # A one-element enum is just the element itself.
    if len(enum.elements) == 1:
        # log.debug("simplified enum: just one element: %s", enum.elements[0])
        return enum.elements[0]

    # log.debug("simplified enum: %s", enum)
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


def maybe_to_string(x):
    try:
        return str(x)
    except:  # noqa: E722
        return '<str failed>'


def append(container, ctor, *args, **kwargs):
    try:
        elt = ctor(*args, **kwargs)
    except NoSignature as e:
        f = e.args[0]
        doc = f.__doc__
        if doc is None or not doc.startswith('Not implemented'):
            qn = qualname(f)
            if qn != '<no name>':
                log.warning(f"no signature for {qn} ({f})")
        return
    except AlreadySeen:
        return
    except (OutsideScope, NoDoc, Deprecated) as e:
        if ctor in [Module, Class]:
            log.warning(
                f"append: caught error building {ctor.__name__} {maybe_to_string(args[0])}: {type(e).__name__}"
            )
        return
    if hasattr(elt, 'ml_name'):
        while elt.ml_name in [getattr(x, 'ml_name', None) for x in container]:
            new_name = elt.ml_name + "'"
            log.warning(
                f"renaming {elt.ml_name} to {new_name} to prevent name clash")
            elt.ml_name = new_name
    container.append(elt)


# XXX TODO make this immutable if possible
class Name(abc.ABC):
    def __init__(self, python_name, parent, ml_name=None):
        assert parent is None or isinstance(
            parent, Name), f"parent is not a Name: {type(parent)}"
        self._parent = parent
        self._python_name = python_name
        if ml_name is None:
            self._ml_name = self.fix_ml(self.python_name())
        else:
            self._ml_name = ml_name
        self._path = list(reversed(self._compute_rev_path()))
        # assert not self.full_ml_name().startswith('Numpy'), self

    @property
    def parent(self):
        return self._parent

    @abc.abstractmethod
    def fix_ml(self, n):
        pass

    def python_name(self):
        return self._python_name

    def ml_name(self):
        return self.fix_ml(self._ml_name)

    def set_ml_name(self, n):
        self._ml_name = n

    def ml_name_opt(self):
        ret = self.ml_name()
        if ret.endswith('_'):
            return ret + 'opt'
        else:
            return ret + '_opt'

    def _compute_rev_path(self):
        ret = []
        x = self
        while x is not None:
            ret.append(x)
            x = x._parent
        return ret

    def full_python_name(self):
        return '.'.join(x.python_name() for x in self._path)

    def full_ml_name(self):
        return '.'.join(x.ml_name() for x in self._path)

    def adjust(self):
        self._ml_name += "'"

    def apply_overrides(self, overrides):
        ml = overrides.ml_name(self.full_python_name())
        if ml is not None:
            self._ml_name = ml

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.full_ml_name()}, {self.full_python_name()})'


class UpperName(Name):
    def fix_ml(self, n):
        return make_module_name(n)


class LowerName(Name):
    def fix_ml(self, n):
        return mlid(n)


# def in_ml_namespace(ns, name):
#     super_ml_name = name.ml_name
#     def ml_name():
#         return ns + '.' + super_ml_name()
#     name.ml_name = ml_name
#     return name


class Package:
    def __init__(self, pkg, overrides, registry, builtin):
        self.pkg = pkg
        self.name = UpperName(self.pkg.__name__, parent=None)
        self.modules = self._list_modules(pkg, overrides, registry, builtin)
        self.registry = registry
        self.overrides = overrides

    def _list_module_raw(self, pkg):
        ret = []
        for mod in pkgutil.iter_modules(pkg.__path__):
            name = mod.name
            if name.startswith('_') or name.endswith('_'):
                continue
            name = UpperName(name, self.name)
            module = importlib.import_module(name.full_python_name())
            ret.append((name, module))
        return ret

    def _list_modules(self, pkg, overrides, registry, builtin):
        ret = []
        modules = self._list_module_raw(pkg)
        # scipy has the toplevel modules (scipy.stats,
        # scipy.linalg...) appearing as submodules in various places,
        # this causes terrible duplication.
        for module in modules:
            overrides.add_blacklisted_submodule(module)
        for name, module in modules:
            append(ret,
                   Module,
                   name,
                   module,
                   overrides=overrides,
                   registry=registry,
                   builtin=builtin)
        return ret

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"Package({self.name.full_python_name()})[\n"
        for mod in self.modules:
            ret += indent(repr(mod) + ",") + "\n"
        ret += "]"
        return ret

    def output_dir(self, path):
        # path / self.pkg.__name__
        return path

    def write(self, path):
        dire = self.output_dir(path)
        log.info(f"writing package {self.name.full_python_name()} to {dire}")
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


class AlreadySeen(Exception):
    pass


class NoDoc(Exception):
    pass


class Deprecated(Exception):
    pass


def write_generated_header(f):
    f.write(
        "(* This file was generated by lib/skdoc.py, do not edit by hand. *)\n"
    )


def fix_naming(elements):
    names = set()
    for elt in elements:
        if hasattr(elt, 'name'):
            while elt.name.ml_name() in names:
                elt.name.adjust()
            names.add(elt.name.ml_name())


class Module:
    def __init__(self, name, module, overrides, registry, builtin, inside=[]):
        assert isinstance(name, UpperName)
        self.name = name
        # Scipy has the modules exported everywhere. Trying to keep only one copy is hard.
        # Reverting to preventing cycles.
        if module in inside:
            raise AlreadySeen(module)
        inside = [module] + inside

        # We pass both the exposed Python name and the builtin module
        # full name through the scope check.
        if not overrides.check_scope(name.full_python_name(), module):
            raise OutsideScope(name)
        if not overrides.check_scope(module.__name__, module):
            raise OutsideScope(name)
        if '.externals.' in name.full_python_name():
            raise OutsideScope(name)

        registry.add(name, module)
        
        self.module = module
        self.elements = self._list_elements(module, overrides, registry,
                                            builtin, inside)
        self.registry = registry
        self.overrides = overrides

    def _list_modules_raw(self, module, overrides):
        modules = []
        for name in dir(module):
            if name.startswith('_') or name.endswith('_'):
                continue
            if name in ['test', 'scipy']:  # for scipy
                continue
            if (name in [
                    'core',
                    'lib',
                    'compat',
                    'ctypeslib',
                    'matrixlib',
                    'scimath',
                    'testing',
                    'records',
            ] and self.name.full_python_name().startswith('numpy')):
                continue
            item = getattr(module, name)
            if overrides.is_blacklisted_submodule(item):
                # print(f"DD blacklisting submodule {qualname}: {item}")
                continue
            if inspect.ismodule(item):
                modules.append((UpperName(name, self.name), item))
        return modules

    def _list_elements(self, module, overrides, registry, builtin, inside):
        # List modules without creating their wrappers, and blacklist
        # them so that if we find them deeper we don't rewrap them.
        modules_raw = self._list_modules_raw(module, overrides)
        for mod in modules_raw:
            overrides.add_blacklisted_submodule(mod)
        modules = []

        # Create classes and functions first so that we find those at
        # the root before the copies in the modules below.
        classes = []
        functions = []
        for name in dir(module):
            # qualname = f"{self.full_python_name}.{name}"
            if name.startswith('_') or name.endswith('_'):
                continue
            if name in ['test', 'scipy']:  # for scipy
                continue
            # emit this one separately, not inside sklearn.metrics
            # if name == "csr_matrix":
            #     continue
            item = getattr(module, name)
            # parent_name = self.full_ml_name
            # print(f"DD wrapping {parent_name}.{name}: {module}")
            if inspect.isclass(item):
                append(classes, Class, UpperName(name, self.name), item,
                       overrides, registry, builtin)
            elif callable(item):
                append(functions, Function, LowerName(name, self.name), item,
                       overrides, registry, builtin)

        # We list modules, and blacklist them so that we don't wrap
        # them again deeper in the hierarchy; then we wrap them.
        for mod_name, mod in modules_raw:
            append(modules, Module, mod_name, mod, overrides, registry,
                   builtin, inside)

        elements = classes + modules + functions
        fix_naming(elements)
        return elements

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"Module({self.name.full_python_name()})[\n"
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
        # Importing the exposed name does not always work, we must
        # import the name of the actual underlying module. See for
        # example numpy.char.
        import_name = self.module.__name__
        f.write(f'let __wrap_namespace = Py.import "{import_name}"\n\n')
        # else:
        #     f.write(
        #         "(* this module has no callables, skipping init and ns *)\n")

    def write(self, path):
        # print(f"DD writing module {self.python_name}")
        # module_path = f"{module_path}.{self.name.ml_name}"
        ml = f"{path / self.name.ml_name()}.ml"
        with open(ml, 'w') as f:
            self.registry.add_generated_file(ml)
            self.write_ml_inside(f)
        mli = f"{path / self.name.ml_name()}.mli"
        with open(mli, 'w') as f:
            self.registry.add_generated_file(mli)
            self.write_mli_inside(f)
            
    def write_ml_inside(self, f):
        self.write_header(f)
        f.write("let get_py name = Py.Module.get __wrap_namespace name\n")
        for element in self.elements:
            element.write_to_ml(f)

    def write_mli_inside(self, f):
        f.write("""(** Get an attribute of this module as a Py.Object.t.
                   This is useful to pass a Python function to another function. *)\n"""
                )
        f.write("val get_py : string -> Py.Object.t\n\n")
        for element in self.elements:
            element.write_to_mli(f)

    def write_doc(self, path):
        md = f"{path / self.name.ml_name()}.md"
        with open(md, 'w') as f:
            self.registry.add_generated_doc_file(md)
            for element in self.elements:
                element.write_to_md(f)

    def write_examples(self, path):
        ml = f"{path / self.name.ml_name()}.ml"
        with open(ml, 'w') as f:
            self.registry.add_generated_file(ml)
            for element in self.elements:
                element.write_examples_to(f)

    def write_to_ml(self, f):
        f.write(f"module {self.name.ml_name()} = struct\n")
        self.write_ml_inside(f)
        f.write("\nend\n")

    def write_to_mli(self, f):
        f.write(f"module {self.name.ml_name()} : sig\n")
        self.write_mli_inside(f)
        f.write("\nend\n\n")

    def write_to_md(self, f):
        full_ml_name = re.sub(r'\.', ".\u200b", self.name.full_ml_name())
        f.write(f"## {self.name.ml_name()}\n\n")
        f.write(f"Module `{full_ml_name}` wraps Python module `{self.name.full_python_name()}`.\n\n")
        for element in self.elements:
            element.write_to_md(f)

    def write_examples_to(self, f):
        full_name = self.name.full_ml_name()
        f.write(f"(*--------- Examples for module {full_name} ----------*)\n")
        for element in self.elements:
            element.write_examples_to(f)


def is_hashable(x):
    try:
        hash(x)
    except TypeError:
        return False
    return True


def caster(x):
    x = re.sub(r'Mixin|Base', '', x)
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    x = re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()
    x = 'as_' + x
    return x


# Major Major
class Class:
    _seen = {}

    def __init__(self, name, klass, overrides, registry, builtin):
        self.name = name
        # print(f"DD wrapping class {parent_name}.{klass.__name__} {klass} {id(klass)}")
        # klass_key = self.name.full_python_name()
        klass_key = klass
        if klass_key in self._seen and len(self._seen[klass_key]) < len(
                self.name.full_python_name()):
            raise AlreadySeen(klass)
        self._seen[klass_key] = self.name.full_python_name()

        if not overrides.check_scope(self.name.full_python_name(), klass):
            raise OutsideScope(self.name)

        registry.add(name, klass)
        
        self.klass = klass

        if inspect.isabstract(klass):
            self.constructor = DummyFunction()
        else:
            try:
                self.constructor = Ctor(LowerName(name.python_name(), name),
                                        self.klass, overrides, registry, builtin)
            except NoSignature:
                log.warning(
                    f"no signature for constructor of {self.name.full_python_name()}, disabling ctor"
                )
                self.constructor = DummyFunction()

        context = TypeContext(klass=self.klass)
        self.elements = self._list_elements(overrides, registry, builtin, context)

        for item in ['deprecated', 'Parallel', 'ABCMeta']:
            if item in self.name.full_python_name():
                raise OutsideScope(klass)

        self.registry = registry
        registry.add_class(klass, self.name.full_python_name())

    def remove_element(self, elt):
        self.elements = [x for x in self.elements if x is not elt]

    def _list_elements(self, overrides, registry, builtin, context):
        elts = []

        # Parse attributes first, so that we avoid warning about them
        # when listing methods.
        attributes = parse_types(str(self.klass.__doc__),
                                 builtin,
                                 context,
                                 section="Attributes")
        attributes = overrides.types(self.name.full_python_name(), attributes,
                                     context.add(group="attributes"))

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
            except (FutureWarning, DeprecationWarning):
                raise Deprecated()
            except Exception:
                # print(f"WW cannot create proto for {self.klass.__name__}: {e}")
                # print(f"WW could not create proto for {self.klass.__name__}")
                proto = self.klass

        for name in dir(proto):
            if name in attributes:
                continue
            # Build our own qualname instead of relying on
            # item.__name__/item.__qualname__, which are unreliable.
            # qualname = f"{self.klass.__name__}.{name}"
            if (name.startswith('_') or name.endswith('_')) and name not in [
                    '__getitem__', '__setitem__', '__iter__'
            ]:
                continue

            try:
                item = getattr(proto, name, None)
            except (FutureWarning, DeprecationWarning):
                # This is deprecated, skip it.
                continue
            except (TypeError, ValueError):
                # Some weird types throw weird errors when you
                # approach their attributes.
                continue

            if item is None:
                continue

            item_name = LowerName(name, self.name)

            # print(f"evaluating possible method {name}: {item}")
            if callable(item):
                if inspect.isclass(item):
                    # print(f"WW skipping class member {item} of proto {proto}")
                    continue
                if is_hashable(item):
                    if item not in callables:
                        append(elts, Method, item_name, item, overrides,
                               registry, builtin)
                        callables.add(item)
                else:
                    append(elts, Method, item_name, item, overrides, registry, builtin)
            elif overrides.has_complete_spec(item_name.full_python_name()):
                # Some methods show up as properties on the class, and
                # are present only in some instantiations (for example
                # predict, inverse_transform on Pipeline). This is a
                # way to have them show up: let overrides specify
                # everything.
                # print(f"II wrapping non-callable method {name}: {item}")
                try:
                    elts.append(Method(item_name, item, overrides, registry, builtin))
                except Exception as e:
                    log.warning(
                        f"method {item_name} has complete override spec but there was an error wrapping it: {e}"
                    )
            elif isinstance(item, property):
                log.warning(f"not wrapping property {item_name}: {item}")
            else:
                pass

        for name, ty in attributes.items():
            append(elts, Attribute, LowerName(name, self.name), ty)
        return elts

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"class({self.name.full_python_name()})[\n"
        ret += indent(repr(self.constructor) + ",") + "\n"
        for elt in self.elements:
            ret += indent(repr(elt) + ",") + "\n"
        ret += "]"
        return ret

    def _tags(self):
        return "[" + ' | '.join(class_tags(self.klass)) + "]"

    def self_tag(self):
        return f"`{tag(self.klass.__name__)}"

    def write_header(self, f):
        f.write(f"type tag = [{self.self_tag()}]\n")
        f.write(f"type t = {self._tags()} Obj.t\n")
        f.write("let of_pyobject x = ((Obj.of_pyobject x) : t)\n")
        f.write("let to_pyobject x = Obj.to_pyobject x\n")
        for base in self.registry.bases[self.klass]:
            base_name = make_module_name(base.__name__)
            f.write(
                f"let {caster(base_name)} x = (x :> [`{tag(base_name)}] Obj.t)\n"
            )

    def write(self, path, ns):
        ml = f"{path / self.name.ml_name()}.ml"
        with open(ml, 'w') as f:
            self.registry.add_generated_file(ml)
            f.write("let () = Wrap_utils.init ();;\n")
            f.write(f'let __wrap_namespace = Py.import "{ns}"\n\n')

            self.write_to_ml(f, wrap=False)

        mli = f"{path / self.name.ml_name()}.mli"
        with open(mli, 'w') as f:
            self.registry.add_generated_file(mli)
            self.write_to_mli(f, wrap=False)

    def write_doc(self, path):
        md = f"{path / self.name.ml_name()}.md"
        with open(md, 'w') as f:
            self.registry.add_generated_doc_file(md)
            self.constructor.write_to_md(f)
            for element in self.elements:
                element.write_to_md(f)

    def write_to_ml(self, f, wrap=True):
        if wrap:
            f.write(f"module {self.name.ml_name()} = struct\n")
        self.write_header(f)
        self.constructor.write_to_ml(f)
        for element in self.elements:
            element.write_to_ml(f)
        f.write(
            "let to_string self = Py.Object.to_string (to_pyobject self)\n")
        f.write('let show self = to_string self\n')
        f.write(
            'let pp formatter self = Format.fprintf formatter "%s" (show self)\n'
        )
        if wrap:
            f.write("\nend\n")

    def write_to_mli(self, f, wrap=True):
        import pdb; pdb.set_trace()  # XXX DEBUG
        gaga
        if wrap:
            f.write(f"module {self.name.ml_name()} : sig\n")
        f.write(f"type tag = [{self.self_tag()}]\n")
        f.write(f"type t = {self._tags()} Obj.t\n")
        f.write("val of_pyobject : Py.Object.t -> t\n")
        f.write("val to_pyobject : [> tag] Obj.t -> Py.Object.t\n\n")
        for base in self.registry.bases[self.klass]:
            base_name = make_module_name(base.__name__)
            # f.write(f"type BaseTypes.{base_name}.t += {base_name} of t\n")
            f.write(
                f"val {caster(base_name)} : t -> [`{tag(base_name)}] Obj.t\n")
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

    def write_to_md(self, f):
        full_ml_name = re.sub(r'\.', ".\u200b", self.name.full_ml_name())
        full_python_name = self.name.full_python_name()
        f.write(f"## {self.name.ml_name()}\n\n")
        if full_python_name.startswith('sklearn'):
            f.write(f"Module `{full_ml_name}` wraps Python class [`{full_python_name}`](https://scikit-learn.org/stable/modules/generated/{full_python_name}.html).\n\n")  # noqa e501
        else:
            f.write(f"Module `{full_ml_name}` wraps Python class `{full_python_name}`.\n\n")
        f.write("~~~ocaml\n")
        f.write("type t\n")
        f.write("~~~\n")
        self.constructor.write_to_md(f)
        for element in self.elements:
            element.write_to_md(f)
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
    if isinstance(t, Optional):
        return t.t

    if not isinstance(t, Enum):
        return t

    def is_none(x):
        return isinstance(x, NoneValue) or (isinstance(x, StringValue)
                                            and x.text == "None")

    none, not_none = partition(t.elements, is_none)
    return simplify_enum(type(t)(not_none))


class Attribute:
    def __init__(self, name, typ):
        assert isinstance(name, Name)
        self.name = name
        self.typ = remove_none_from_enum(typ)

    def iter_types(self):
        yield self.typ

    def write_to_ml(self, f):
        unwrap = _localize(self.typ.unwrap, self.name.parent)
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
let {self.name.ml_name_opt()} self =
  match Py.Object.get_attr_string (to_pyobject self) "{self.name.python_name()}" with
  | None -> failwith "attribute {self.name.python_name()} not found"
  | Some x -> if Py.is_none x then None else Some ({unwrap} x)

let {self.name.ml_name()} self = match {self.name.ml_name_opt()} self with
  | None -> raise Not_found
  | Some x -> x
""")

    def write_to_mli(self, f):
        #  XXX TODO extract doc and put it here
        ml_type_ret = _localize(self.typ.ml_type_ret, self.name.parent)
        f.write(f"""
(** Attribute {self.name.python_name()}: get value or raise Not_found if None.*)
val {self.name.ml_name()} : t -> {ml_type_ret}

(** Attribute {self.name.python_name()}: get value as an option. *)
val {self.name.ml_name_opt()} : t -> ({ml_type_ret}) option

""")

    def write_to_md(self, f):
        ml_type_ret = _localize(self.typ.ml_type_ret, self.name.parent)
        f.write(f"""
### {self.name.ml_name()}

???+ note "attribute"
    ~~~ocaml
    val {self.name.ml_name()} : t -> {ml_type_ret}
    val {self.name.ml_name_opt()} : t -> ({ml_type_ret}) option
    ~~~

    This attribute is documented in `create` above. The first version raises Not_found
    if the attribute is None. The _opt version returns an option.
""")

    def write_examples_to(self, f):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Attribute({self.name.full_ml_name()} : {self.typ})"


def clean_doc(doc):
    if doc is None:
        return doc
    doc = re.sub(r'\*\)', '* )', str(doc))
    doc = re.sub(r'\(\*', '( *', doc)
    doc = re.sub(r'\{\|', '{ |', doc)
    doc = re.sub(r'\|\}', '| }', doc)
    # just do away with double quotes entirely
    doc = re.sub(r'"', "'", doc)
    # num_quotes = doc.count('"')
    # if num_quotes % 2 == 1:
    #     doc = doc + '"'
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
                 r'~~~python\n\1\n~~~\n\n',
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
            klass = make_module_name(self.env.get(obj, ''))
            method = mlid(m.group(2))
            ml_args = format_ml_args(m.group(3))
            return f"{klass}.{method} {ml_args} {obj}"

        def replace_attribute(m):
            obj = m.group(1)
            klass = make_module_name(self.env.get(obj, ''))
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
        module = make_module_name(
            os.path.splitext(os.path.split(f.name)[-1])[0])
        f.write("(* TEST TODO\n")
        f.write(f'let%expect_test "{self.name}" =\n')
        f.write(f"  let open Sklearn.{module} in\n")
        for element in self.elements:
            element.write(Indented(f))
        f.write("\n*)\n\n")


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

    def __call__(self, context):
        """This is so that users can put either a Type or a function context
        -> Type in overrides.

        """
        return self

    names = []

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
        t_ret = f"`{ta} of {self.ml_type_ret}"
        destruct = f"`{ta} x -> {self.wrap} x"
        if self.is_type is None:
            construct = None
        else:
            construct = f"if {self.is_type} x then `{ta} ({self.unwrap} x)"
        return t, t_ret, destruct, construct

    def delegate_to(self, t):
        self.ml_type = t.ml_type
        self.ml_type_ret = t.ml_type_ret
        self.wrap = t.wrap
        self.unwrap = t.unwrap
        self.tag = t.tag
        self.tag_name = t.tag_name

    def __hash__(self):
        hash_me = {}
        for k, v in vars(self).items():
            if isinstance(v, list):
                v = tuple(v)
            hash_me[k] = v
        return hash((self.__class__, tuple(hash_me.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Ignored(Type):
    names = ['ignored', 'Ignored']


class RandomState(Type):
    names = ['RandomState instance', 'RandomState', 'instance of RandomState']


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
        t_ret = t
        destruct = f'`{ta} -> {self.wrap} "{self.text}"'
        construct = None
        return t, t_ret, destruct, construct


class IntValue(Type):
    def __init__(self, value):
        self.value = int(value)

    ml_type = 'int'
    wrap = 'Py.Int.of_int'

    ml_type_ret = 'int'
    unwrap = 'Py.Int.to_int'

    def tag(self):
        tags = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
        assert 0 <= self.value < len(tags), (self.value, tags)
        ta = tags[self.value]
        t = f"`{ta}"
        t_ret = t
        destruct = f'`{ta} -> {self.wrap} {self.value}'
        construct = None
        return t, t_ret, destruct, construct

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class Enum(Type):
    def can_be_returned(self):
        for elt in self.elements:
            if elt.tag()[3] is None:
                # print(f"WW enum {self} cannot be returned because element {elt} cannot be discriminated")
                return False
        return True

    def __init__(self, elements, name=None):
        # print(f"DD building enum: {elements}")
        for elt in elements:
            assert isinstance(elt, Type), elt
        assert elements, elements
        self.elements = elements
        self.ml_type = "[" + ' | '.join([elt.tag()[0]
                                         for elt in elements]) + "]"
        wrap_cases = [f"| {elt.tag()[2]}\n" for elt in elements]
        self.wrap = f"(function\n{''.join(wrap_cases)})"

        if self.can_be_returned():
            self.ml_type_ret = "[" + ' | '.join(
                [elt.tag()[1] for elt in elements]) + "]"
            unwrap_cases = [f"{elt.tag()[3]}" for elt in elements]
            self.unwrap = '(fun x -> ' + ' else '.join(
                unwrap_cases
            ) + ''' else failwith (Printf.sprintf "Sklearn: could not identify type from Python value %s (%s)"
                                                  (Py.Object.to_string x) (Wrap_utils.type_string x)))'''

        if name is not None:
            self.tag_name = lambda: name


class Optional(Type):
    def __init__(self, t):
        # args are wrapped as a enum including `None
        # returns are wrapped using an option
        self.t = t
        if isinstance(t, Enum):
            self.as_enum = Enum(t.elements + [NoneValue()])
        else:
            self.as_enum = Enum([t, NoneValue()])
        self.ml_type = self.as_enum.ml_type
        self.wrap = self.as_enum.wrap
        self.ml_type_ret = f'{self.t.ml_type_ret} option'
        self.unwrap = f'(fun py -> if Py.is_none py then None else Some ({self.t.unwrap} py))'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t})'


class Tuple(Type):
    def __init__(self, elements, names=None):
        if names is not None:
            self.names = names
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
        self.is_type = 'Py.Tuple.check'


class Int(Type):
    names = [
        'int', 'integer', 'integer > 0', 'int with', 'int > 1',
        'strictly positive integer', 'an integer'
    ]
    ml_type = 'int'
    wrap = 'Py.Int.of_int'
    ml_type_ret = 'int'
    unwrap = 'Py.Int.to_int'
    is_type = 'Wrap_utils.check_int'

    def tag_name(self):
        return tag("I")


class Unit(Type):
    names = []
    ml_type = 'unit'
    wrap = '(fun () -> Py.none)'
    ml_type_ret = 'unit'
    unwrap = '(fun _ -> ())'
    # is_type = 'Wrap_utils.check_int'
    # def tag_name(self):
    #     return tag("I")


class WrappedModule(Type):
    def __init__(self, module, names=None):
        if names is not None:
            self.names = names
        self.ml_type = f'{module}.t'
        self.ml_type_ret = f'{module}.t'
        self.wrap = f'{module}.to_pyobject'
        self.unwrap = f'{module}.of_pyobject'


class Float(Type):
    names = [
        'float', 'floating', 'double', 'positive float',
        'strictly positive float', 'non-negative float',
        'float (upperlimited by 1.0)', 'float in range', 'float in [0., 1.]',
        'float in', 'float between 0 and 1', '0 <= shrinkage <= 1',
        '0 < support_fraction < 1', 'non-negative real', '0 < double < 1',
        'float between 0.0 and 1.0'
    ]
    ml_type = 'float'
    wrap = 'Py.Float.of_float'
    ml_type_ret = 'float'
    unwrap = 'Py.Float.to_float'
    is_type = 'Wrap_utils.check_float'

    def tag_name(self):
        return tag("F")


class Bool(Type):
    names = [
        'bool', 'boolean', 'Boolean', 'Bool', 'boolean value', 'type boolean'
    ]
    ml_type = 'bool'
    wrap = 'Py.Bool.of_bool'
    ml_type_ret = 'bool'
    unwrap = 'Py.Bool.to_bool'


class Generator(Type):
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.ml_type = f'{t.ml_type} Seq.t'
        self.wrap = f'(fun ml -> Seq.map {t.wrap} ml |> Py.Iter.of_seq)'
        self.ml_type_ret = f'{t.ml_type_ret} Seq.t'
        self.unwrap = f'(fun py -> Py.Iter.to_seq py |> Seq.map {t.unwrap})'
        self.is_type = 'Py.Iter.check'

    def tag_name(self):
        return 'Iter'


def ArrGenerator():
    ret = Generator(Arr())
    ret.names = ['generator of array']
    return ret


class LossFunction(Type):
    names = ['LossFunction', 'concrete ``LossFunction``']
    # ml_type = 'Sklearn.Arr.t -> Sklearn.Arr.t -> float'
    ml_type_ret = 'Np.NumpyRaw.Ndarray.t -> Np.NumpyRaw.Ndarray.t -> float'
    unwrap = '''(fun py -> fun x y -> Py.Callable.to_function py
       [|Np.NumpyRaw.Ndarray.to_pyobject x; Np.NumpyRaw.Ndarray.to_pyobject y|] |> Py.Float.to_float)'''


class ClassificationReport(Type):
    # ml_type = '(string * <precision:float; recall:float; f1_score:float; support:float>) list'
    ml_type_ret = '(string * <precision:float; recall:float; f1_score:float; support:float>) list'
    unwrap = '''(fun py -> Py.Dict.fold (fun kpy vpy acc -> ((Py.String.to_string kpy), object
      method precision = Py.Dict.get_item_string vpy "precision" |> Wrap_utils.Option.get |> Py.Float.to_float
      method recall = Py.Dict.get_item_string vpy "recall" |> Wrap_utils.Option.get |> Py.Float.to_float
      method f1_score = Py.Dict.get_item_string vpy "f1-score" |> Wrap_utils.Option.get |> Py.Float.to_float
      method support = Py.Dict.get_item_string vpy "support" |> Wrap_utils.Option.get |> Py.Float.to_float
    end)::acc) py [])
    '''
    is_type = 'Py.Dict.check'

    def tag_name(self):
        return 'Dict'


class Array(Type):
    def __init__(self, t, names=[]):
        self.t = t
        self.names = names
        self.ml_type = f"{t.ml_type} array"
        self.wrap = f'(fun ml -> Py.Array.of_array {t.wrap} (fun _ -> invalid_arg "read-only") ml)'
        self.ml_type_ret = f"{t.ml_type} array"
        self.unwrap = f"""(fun py -> let len = Py.Sequence.length py in Array.init len
          (fun i -> {t.unwrap} (Py.Sequence.get_item py i)))"""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Array[{str(self.t)}]"


class ArrPyList(WrappedModule):
    def __init__(self):
        super().__init__('Np.Numpy.Ndarray.List',
                         names=[
                             'iterable of iterables', 'list of arrays',
                             'list of ndarray', 'arrays', 'list of array-like'
                         ])


class SparsePyList(WrappedModule):
    def __init__(self):
        super().__init__('Sklearn.SparseMatrixList',
                         names=[
                             'sparse matrices',
                         ])


class List(Type):
    def __init__(self, t, names=None):
        self.t = t
        if names is not None:
            self.names = names
        self.ml_type = f"{t.ml_type} list"
        self.wrap = f'(fun ml -> Py.List.of_list_map {t.wrap} ml)'
        self.ml_type_ret = f"{t.ml_type_ret} list"
        self.unwrap = f"(fun py -> Py.List.to_list_map ({t.unwrap}) py)"

    def tag_name(self):
        return self.t.tag_name() + 's'

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"List[{str(self.t)}]"


class ListAsTuple(Type):
    """On the ml side, a list. On the Python side, a tuple.
    """
    def __init__(self, t, names=None):
        self.t = t
        if names is not None:
            self.names = names
        self.ml_type = f"{t.ml_type} list"
        self.wrap = f'(fun ml -> Py.Tuple.of_list_map {t.wrap} ml)'
        self.ml_type_ret = f"{t.ml_type_ret} list"
        self.unwrap = f"(fun py -> Py.Tuple.to_list_map ({t.unwrap}) py)"

    def tag_name(self):
        return self.t.tag_name() + 's'

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"ListAsTuple[{str(self.t)}]"


class StarStar(Type):
    def __init__(self):
        self.names = []
        self.ml_type = "(string * Py.Object.t) list"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "**kwargs"


class FloatList(List):
    def __init__(self):
        super().__init__(
            Float(),
            names=['list of floats', 'list positive floats', 'list of float'])


class StringList(Type):
    names = [
        'list of strings', 'list of string', 'list/tuple of strings',
        'tuple of str', 'list of str', 'strings', 'tuple of strings'
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
    is_type = 'Py.String.check'

    def tag_name(self):
        return tag("S")


class Scalar(Enum):
    names = ['scalar']

    def __init__(self, elements=[Float(), Int(), Bool(), String()]):
        super().__init__(elements)


class Number(Enum):
    names = ['number', 'numeric', 'numerical value']

    def __init__(self, elements=[Float(), Int()]):
        super().__init__(elements)


class NoneValue(Type):
    names = ['None', 'none']

    def tag(self):
        return '`None', '`None', '`None -> Py.none', None


class TrueValue(Type):
    names = ['True', 'true']

    def tag(self):
        return '`True', '`True', '`True -> Py.Bool.t', None


class FalseValue(Type):
    names = ['False', 'false']

    def tag(self):
        return '`False', '`False', '`False -> Py.Bool.f', None


def interesting_ancestors(klass):
    ancestors = inspect.getmro(klass)
    for ancestor in ancestors:
        if ancestor is not klass:
            ancestor_name = ancestor.__name__.lower()
            # tweaked for sklearn+scipy
            if (('base' in ancestor_name or 'mixin' in ancestor_name
                 or 'rv_' in ancestor_name or 'decisiontree' in ancestor_name)
                    and not ancestor_name.startswith('_')):
                # if not ancestor_name.startswith('_'):
                yield ancestor


def class_tags(klass):
    import scipy
    import numpy as np
    ancestors = interesting_ancestors(klass)
    tags = set(f"`{tag(base.__name__)}"
               for base in (list(ancestors) + [klass]))
    tags.add("`Object")
    if set([scipy.sparse.spmatrix,
            np.ndarray]).intersection(inspect.getmro(klass)):
        tags.add("`ArrayLike")
    return sorted(tags)


class BaseTypeByTag(Type):
    def __init__(self, tag, inside_np=False):
        if inside_np:
            obj = 'Obj'
        else:
            obj = 'Np.Obj'
        self._tag_name = tag
        self.ml_type = f'[>`{self._tag_name}] {obj}.t'
        self.wrap = f'{obj}.to_pyobject'
        self.ml_type_ret = f'[>`{self._tag_name}] {obj}.t'
        self.unwrap = f'(fun py -> ({obj}.of_pyobject py : {self.ml_type_ret}))'

    def tag_name(self):
        return self._tag_name

    def __repr__(self):
        return f'{self.__class__.__name__}({self._tag_name})'


class BaseType(Type):
    def __init__(self, name, inside_np=False):
        if inside_np:
            obj = 'Obj'
        else:
            obj = 'Np.Obj'
        self.name = name
        self._tag_name = tag(name.split('.')[-1])
        self.ml_type = f'[>`{self._tag_name}] {obj}.t'
        self.wrap = f'{obj}.to_pyobject'
        mo = '.'.join(name.split('.')[:-1])
        # log.debug(f"import module '{mo}'")
        module = importlib.import_module(mo)
        # log.debug(f"eval '{name}'")
        klass_name = name.split('.')[-1]
        klass = getattr(module, klass_name)
        # klass = eval(name)
        tags = class_tags(klass)
        self.ml_type_ret = f'[{"|".join(tags)}] {obj}.t'
        self.unwrap = f'(fun py -> ({obj}.of_pyobject py : {self.ml_type_ret}))'

    def tag_name(self):
        return self._tag_name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


class CsrMatrix(BaseType):
    names = [
        'CSR matrix with',
        'CSR matrix',
        'CSR sparse matrix',
        'CSR',
        'scipy csr array',
        'scipy.sparse.csr_matrix of floats'
        'scipy.sparse.csr_matrix',
        'sparse (CSR) matrix',
        'sparse CSR matrix',
        'sparse graph in CSR format',
    ]

    def __init__(self):
        import scipy.sparse  # noqa f401
        super().__init__('scipy.sparse.csr_matrix')


class CscMatrix(BaseType):
    names = [
        'CSC matrix with',
        'CSC matrix',
        'CSC sparse matrix',
        'CSC',
        'scipy csc array',
        'scipy.sparse.csc_matrix of floats'
        'scipy.sparse.csc_matrix',
        'sparse (CSC) matrix',
        'sparse CSC matrix',
        'sparse graph in CSC format',
    ]

    def __init__(self):
        import scipy.sparse  # noqa f401
        super().__init__('scipy.sparse.csc_matrix')


class SparseMatrix(BaseType):
    names = [
        # the sparse matrices
        '(sparse) array-like',
        'scipy.sparse matrix',
        'scipy.sparse',
        'sparse array',
        'sparse matrix with',
        'sparse matrix',
        'sparse-matrix',
    ]

    def __init__(self):
        import scipy.sparse  # noqa f401
        super().__init__('scipy.sparse.spmatrix')


class Arr(BaseTypeByTag):
    names = [
        '1D array',
        '1d array-like',
        '2D array',
        '2D ndarray',
        '2D numpy.ndarray',
        '3D array',
        'A collection of strings',
        'an iterable',
        'array  or',
        'array (n_samples]',
        'array [n_core_samples]',
        'array like',
        'array of float',
        'array of floats',
        'array of int',
        'array of integers',
        'array of shape `shape`'
        'array of shape `shape`',
        'array of shape of (n_targets',
        'array',
        'array-like of float',
        'array-like of floats',
        'array-like of shape at least 2D',
        'array-like of shape',
        'array-like',
        'array_like of float',
        'array_like of floats',
        'array_like',
        'bool array',
        'collection of string',
        'float array with',
        'float ndarray',
        'indexable',
        'int array',
        'int array-like',
        'integer ndarray',
        'iterable',
        'label indicator array / sparse matrix',
        'label indicator matrix',
        'list',
        'list-like',
        'matrix',
        'nd-array',
        'ndarray of floats',
        'ndarray of reals',
        'ndarray or scalar',
        'ndarray',
        '{ndarray}',
        'ndarray, int',
        'ndarray, float',
        'ndarray, bool',
        'ndarray, see dtype parameter above',
        'np.array',
        'np.matrix',
        'numeric array-like',
        'numpy array . Shape depends on ``subset``',
        'numpy array of float',
        'numpy array of int',
        'numpy array',
        'numpy.matrix',
        'numpy.ndarray',
        'record array',
        'sequence of floats',
        '{array}',
    ]

    def __init__(self, inside_np=False):
        super().__init__('ArrayLike', inside_np)

    def tag_name(self):
        return 'Arr'


class Ndarray(BaseType):
    names = Arr.names

    def __init__(self, inside_np=False):
        import numpy  # noqa f401
        super().__init__('numpy.ndarray', inside_np)


class CrossValGenerator(BaseType):
    names = [
        'cross-validation generator', 'cross-validation splitter',
        'CV splitter', 'a cross-validator instance'
    ]

    def __init__(self, inside_np=False):
        import sklearn.model_selection  # noqa f401
        super().__init__('sklearn.model_selection.BaseCrossValidator',
                         inside_np)


class Estimator(BaseType):
    names = [
        'instance BaseEstimator', 'BaseEstimator instance', 'BaseEstimator',
        'estimator instance', 'instance estimator', 'estimator object',
        'estimator'
    ]

    def __init__(self, inside_np=False):
        import sklearn.base  # noqa f401
        super().__init__('sklearn.base.BaseEstimator', inside_np)


class Regressor(BaseType):
    names = [
        'instance RegressorMixin', 'RegressorMixin instance',
        'regressor instance', 'instance regressor', 'regressor object',
        'regressor'
    ]

    def __init__(self, inside_np=False):
        import sklearn.base  # noqa f401
        super().__init__('sklearn.base.RegressorMixin', inside_np)


class Transformer(BaseType):
    names = [
        'instance TransformerMixin', 'TransformerMixin instance',
        'instance Transformer', 'Transformer instance', 'transformer instance',
        'instance transformer', 'transformer object', 'transformer'
    ]

    def __init__(self, inside_np=False):
        import sklearn.base  # noqa f401
        super().__init__('sklearn.base.TransformerMixin', inside_np)


class ClusterEstimator(BaseType):
    names = ['instance of sklearn.cluster model', 'sklearn.cluster model']

    def __init__(self, inside_np=False):
        import sklearn.base  # noqa f401
        super().__init__('sklearn.base.ClusterMixin', inside_np)


class DecisionTreeClassifier(BaseType):
    names = ['decision tree classifier']

    def __init__(self, inside_np=False):
        import sklearn.tree  # noqa f401
        super().__init__('sklearn.tree.DecisionTreeClassifier', inside_np)


class PyObject(Type):
    names = ['object']


class Self(Type):
    def __init__(self):
        self.ml_type = '[> tag] Obj.t'
        self.ml_type_ret = 't'
        self.wrap = 'to_pyobject'
        self.unwrap = 'of_pyobject'


class Dtype(WrappedModule):
    names = [
        'type', 'dtype', 'numpy dtype', 'data-type', 'column dtype',
        'numpy data type'
    ]

    def __init__(self, inside_np=False):
        if inside_np:
            super().__init__('Dtype')
        else:
            super().__init__('Np.Dtype')
        # ml_type = 'Sklearn.Arr.Dtype.t'
        # wrap = 'Sklearn.Arr.Dtype.to_pyobject'
        # ml_type_ret = 'Sklearn.Arr.Dtype.t'
        # unwrap = 'Sklearn.Arr.Dtype.of_pyobject'


class Dict(Type):
    # XXX 'dict of numpy (masked) ndarrays' is the format of a
    # DataFrame, we could have a more appropriate OCaml type for it
    # maybe, rather than this generic one (which is just a Python
    # dict)?
    names = [
        'dict', 'Dict', 'dictionary', 'mapping of string to any',
        'dict of numpy (masked) ndarrays', 'dict of numpy ndarrays',
        'dict of float arrays', 'dictionary of string to any',
        'dict of string -> object', 'a dict'
    ]
    ml_type = 'Sklearn.Dict.t'
    ml_type_ret = 'Sklearn.Dict.t'
    wrap = 'Sklearn.Dict.to_pyobject'
    unwrap = 'Sklearn.Dict.of_pyobject'
    is_type = 'Py.Dict.check'


class ParamGridDict(Type):
    names = []
    ml_type = '''(string * Sklearn.Dict.param_grid) list'''
    wrap = '(fun x -> Sklearn.Dict.(of_param_grid_alist x |> to_pyobject))'

    def tag_name(self):
        return 'Grid'


class ParamDistributionsDict(Type):
    names = []
    ml_type = '''(string * Sklearn.Dict.param_distributions) list'''
    wrap = '(fun x -> Sklearn.Dict.(of_param_distributions_alist x |> to_pyobject))'

    def tag_name(self):
        return 'Grid'


# emitted as a postprocessing of Dict() based on param name/callable
class DictIntToFloat(Type):
    names = ['dict int to float']
    ml_type = '(int * float) list'
    wrap = '(Py.Dict.of_bindings_map Py.Int.of_int Py.Float.of_float)'
    is_type = 'Py.Dict.check'


class Callable(Type):
    names = ['callable', 'function']


class Slice(Type):
    names = ['slice']

    def __init__(self, inside_np=False):
        if inside_np:
            ns = "Wrap_utils"
        else:
            ns = "Np.Wrap_utils"
        self.ml_type = f'{ns}.Slice.t'
        self.ml_type_ret = f'{ns}.Slice.t'
        self.wrap = f'{ns}.Slice.to_pyobject'
        self.unwrap = f'{ns}.Slice.of_pyobject'


estimator_alist = List(
    Tuple([String(), Estimator()]),
    names=['list of (str, estimator)', 'list of (str, estimator) tuples'])


class BuiltinTypes:
    def __init__(self, builtins):
        self._builtins = {}
        for t in builtins:
            for name in t.names:
                assert name not in self._builtins, f"'{name}' is present several times in builtin types"
                self._builtins[name] = t
        self._used_names = set()

    def __getitem__(self, k):
        self._used_names.add(k)
        return self._builtins[k]

    def report_unused(self):
        unused = set(self._builtins.keys()).diff(self._used_names)
        if unused:
            log.warning("some type names were not used:")
            for name in unused:
                log.warning("- {name} ({self._builtins[name]})")


class TupleOfInts(Type):
    """A tuple of ints is exposed as int list on input, and int array on
    output (more convenient for shapes).

    """
    names = ['tuple of ints']
    li = ListAsTuple(Int())
    arr = Array(Int())
    wrap = li.wrap
    ml_type = li.ml_type
    unwrap = arr.unwrap
    ml_type_ret = arr.ml_type_ret


generic_builtin_types = [
    Int(),
    Float(),
    Bool(),
    Scalar(),
    Number(),
    FloatList(),
    StringList(),
    SparseMatrix(),
    String(),
    DictIntToFloat(),
    NoneValue(),
    TrueValue(),
    FalseValue(),
    PyObject(),
    Self(),
    Callable(),
    Ignored(),
    List(Int(), names=['sequence of ints']),
    Tuple([Int(), Int()], names=['(int, int)']),
    Tuple([Float(), Float()],
          names=['pair of floats', '(float, float)', 'pair of floats >= 0']),
    List(Tuple([Int(), Int()]), names=['list of (int, int)']),
    RandomState(),
    TupleOfInts(),
]

numpy_builtin_types = BuiltinTypes(generic_builtin_types + [
    Ndarray(inside_np=True),
    Dtype(inside_np=True),
    Slice(inside_np=True),
    List(Ndarray(inside_np=True), names=['sequence of ndarrays'])
])

scipy_builtin_types = BuiltinTypes(generic_builtin_types + [
    Ndarray(),
    Dtype(),
    Slice(),
    List(Ndarray(), names=['sequence of ndarrays'])
])

sklearn_builtin_types = BuiltinTypes(generic_builtin_types + [
    Arr(),
    ArrGenerator(),
    LossFunction(),
    ArrPyList(),
    SparsePyList(),
    Dict(),
    Dtype(),
    Slice(),
    List(Dtype(), names=['list of column dtypes', 'list of types']),
    CrossValGenerator(),
    Estimator(),
    ClusterEstimator(),
    DecisionTreeClassifier(),
    WrappedModule('Sklearn.Pipeline.Pipeline', ['Pipeline', 'pipeline']),
    WrappedModule('Sklearn.Pipeline.FeatureUnion', ['FeatureUnion']),
    estimator_alist,
    List(Arr(), names=['list of numpy arrays']),
    List(Estimator(), names=['list of estimators', 'list of estimator']),
    List(Regressor(), names=['list of regressors', 'list of regressor']),
    List(Tuple([String(), Transformer()]),
         names=['list of (string, transformer) tuples']),
    Tuple([Arr(), Arr()], names=['tuple of (A, B) ndarrays']),
    CsrMatrix(),
    CscMatrix()
])


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
    if '__is_yield' in type_dict:
        is_yield = True
        del type_dict['__is_yield']
    else:
        is_yield = False

    if not type_dict:
        return PyObject()

    if len(type_dict) == 1:
        if list(type_dict.keys())[0] == 'self':
            return Self()
        v = list(type_dict.values())[0]
        if is_yield:
            if isinstance(v, Arr):
                return ArrGenerator()
            elif not isinstance(v, Generator) and not isinstance(v, PyObject):
                log.warning(
                    f"yielded value is not a generator ({v}), forcing type to Py.Object.t"
                )
                return PyObject()
            else:
                return v
        else:
            return v

    assert len(type_dict) > 1, type_dict

    if is_yield:
        for k, ty in type_dict.items():
            assert not isinstance(
                ty, Generator
            ), f"yielded tuple has a suspect Generator: {type_dict}"
        return Generator(Tuple(list(type_dict.values())))

    return Tuple(list(type_dict.values()))


class NoSignature(Exception):
    pass


def _localize(t, parent_name):
    """Attempt to localize a type or expression by removing the necessary
    parts of paths. Working on strings is inherently flawed, the right
    way to do this would be to pass the module path to the things that
    generate t, so that they can generate clean types and
    expressions. Anyway, seems to work in practice.

    TODO: move this to Name?
    """
    if parent_name is None:
        return t

    path_elts = parent_name.full_ml_name().split('.')
    ret = t

    # assert not parent_name.full_ml_name().startswith('Numpy'), parent_name

    for i in range(len(path_elts), 0, -1):
        path = r'\b' + r'\.'.join(path_elts[:i]) + r'\.'
        ret = re.sub(path, '', ret)

    # if "Csr_matrix" in t:
    # if 'Np' in t:
    # print(f"DD localize: {t} / {parent_name} -> {ret}")
    return ret


class Parameter:
    def __init__(self, name, ty, parameter):
        self.name = name
        self.ty = ty
        self.parameter = parameter
        self.fixed_value = None
        self.no_name = False
        # If the default value is None, no need to have it as an option in the type.
        if self.parameter.default is None:
            self.ty = remove_none_from_enum(self.ty)

    def remove_name(self):
        if self.has_default():
            log.warning(
                "cannot remove name for param {self.name.full_python_name()}: param has a default"
            )
            return
        self.no_name = True

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

        return f"{mark}{self.name.ml_name()}{default}{fixed} : {self.ty}"

    def _has_fixed_value(self):
        return self.fixed_value is not None

    def is_star(self):
        return self.parameter.kind == inspect.Parameter.VAR_POSITIONAL

    def is_star_star(self):
        return self.parameter.kind == inspect.Parameter.VAR_KEYWORD

    def is_positional_only(self):
        return self.parameter.kind == inspect.Parameter.POSITIONAL_ONLY

    def is_named(self):
        if self.no_name:
            return False
        if self.has_default():
            return True
        # if self.is_positional_only():
        #     return False
        return not (self.name.ml_name() in ['self'] or self.is_star())

    def has_default(self):
        return ((self.parameter.default is not inspect.Parameter.empty)
                or self.is_star_star())

    def sig(self):
        if self._has_fixed_value():
            return None
        ml_type = _localize(self.ty.ml_type, self.name.parent)
        if self.is_named():
            spec = f"{self.name.ml_name()}:{ml_type}"
            if self.has_default():
                spec = f"?{spec}"
            return spec
        else:
            return ml_type

    def decl(self):
        if self._has_fixed_value():
            return None
        spec = f"{self.name.ml_name()}"
        if self.has_default():
            spec = f"?{spec}"
        elif self.is_named():
            spec = f"~{spec}"
        return spec

    def call(self):
        """Return a tuple:
        - None or (name, value getter)
        - None or *args param name
        - None or **kwargs param name
        """

        ty_wrap = _localize(self.ty.wrap, self.name.parent)

        if ty_wrap == 'Wrap_utils.id':
            pipe_ty_wrap = ''
        else:
            pipe_ty_wrap = f'|> {ty_wrap}'

        if self._has_fixed_value():
            wrap = f'Some({self.fixed_value} {pipe_ty_wrap})'
        elif self.has_default():
            if ty_wrap == 'Wrap_utils.id':
                wrap = self.name.ml_name()
            else:
                wrap = f'Wrap_utils.Option.map {self.name.ml_name()} {ty_wrap}'
        else:
            wrap = f'Some({self.name.ml_name()} {pipe_ty_wrap})'

        kv = (self.name.python_name(), wrap)

        pos_arg = None
        if self.is_star():
            ty_t_wrap = _localize(self.ty.t.wrap, self.name.parent)
            pos_arg = f"(List.map {ty_t_wrap} {self.name.ml_name()})"
            # pos_arg = f"(Wrap_utils.pos_arg {ty_t_wrap} {self.name.ml_name()})"
            kv = None

        if self.is_positional_only():
            # assert not self.has_default(), self
            assert not self._has_fixed_value(), self
            if self.has_default():
                # XXX this is dangerous, we should probably assert
                # somewhere that if this is None there is no other
                # positional arg coming afer it (at runtime ?)
                pos_arg = f"(match {self.name.ml_name()} with None -> [] | Some x -> [x {pipe_ty_wrap}])"
            else:
                pos_arg = f"[{self.name.ml_name()} {pipe_ty_wrap}]"
            kv = None

        kw_arg = None
        if self.is_star_star():
            kw_arg = f"(match {self.name.ml_name()} with None -> [] | Some x -> x)"
            kv = None

        return kv, pos_arg, kw_arg


class DummyUnitParameter:
    python_name = ''
    ty = None

    def is_named(self):
        return False

    def remove_name(self):
        pass

    def has_default(self):
        return False

    def sig(self):
        return 'unit'

    def decl(self):
        return '()'

    def call(self):
        return None, None, None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '()'


class SelfParameter:
    def __init__(self):
        # self.python_name = 'self'
        self.ty = Self()

    def remove_name(self):
        pass

    def is_named(self):
        return False

    def has_default(self):
        return False

    def sig(self):
        return self.ty.ml_type

    def decl(self):
        return 'self'

    def call(self):
        return None, None, None

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.ty)


class Return:
    def __init__(self, ty, name):
        self.name = name
        self.ty = ty

    def sig(self):
        return _localize(self.ty.ml_type_ret, self.name.parent)

    def call(self):
        return _localize(self.ty.unwrap, self.name.parent)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.ty)


class Wrapper:
    def __init__(self, name, doc, parameters, ret, doc_type, namespace):
        self.name = name
        self.parameters = self._fix_parameters(parameters)
        self.ret = ret
        self.doc = clean_doc(doc)
        self.doc_type = doc_type
        self.namespace = namespace

    def _fix_parameters(self, parameters):
        """Put not-named parameters at the end, and add a dummy unit parameter
        at the end if needed.

        """
        if len(parameters) == 1 and not parameters[0].has_default():
            parameters[0].remove_name()

        named, not_named = partition(parameters, lambda x: x.is_named())
        with_default, no_default = partition(named, lambda x: x.has_default())
        parameters = with_default + no_default + not_named
        if not not_named:
            # we have only named params
            parameters.append(DummyUnitParameter())
        return parameters

    def _join_not_none(self, sep, elements):
        elements = filter(lambda x: x is not None, elements)
        return sep.join(elements)

    def mli(self, doc=True):
        sig = self._join_not_none(' -> ', [x.sig() for x in self.parameters] +
                                  [self.ret.sig()])
        if doc:
            return f"val {self.name.ml_name()} : {sig}\n(**\n{self.doc}\n*)\n\n"
        else:
            return f"val {self.name.ml_name()} : {sig}\n"

    def md(self):
        sig = self._join_not_none(" ->\n", [x.sig() for x in self.parameters] +
                                  [self.ret.sig()])
        sig = indent(sig, 6)
        if self.doc is None:
            doc = ''
        else:
            doc = format_md_doc(self.doc)
        doc = indent(doc)
        return f"""
### {self.name.ml_name()}

???+ note "{self.doc_type}"
    ~~~ocaml
    val {self.name.ml_name()} :
{sig}
    ~~~

{doc}
"""

    def _pos_args(self):
        pos_args = []
        for param in self.parameters:
            kv, pos, kw = param.call()
            if pos is not None:
                pos_args.append(pos)

        if not pos_args:
            return '[||]'
        else:
            return f'(Array.of_list @@ List.concat [{";".join(pos_args)}])'

    def _kw_args(self):
        pairs = []
        kwargs = None
        for param in self.parameters:
            kv, _pos, kw = param.call()
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

    def ml(self):
        arg_decl = self._join_not_none(' ',
                                       [x.decl() for x in self.parameters])
        pos_args = self._pos_args()
        kw_args = self._kw_args()

        ret_call = self.ret.call()
        if ret_call == 'Wrap_utils.id':
            pipe_ret_call = ''
        else:
            pipe_ret_call = f'|> {ret_call}'

        ret = f"""\
                  let {self.name.ml_name()} {arg_decl} =
                     Py.Module.get_function_with_keywords {self.namespace} "{self.name.python_name()}"
                       {pos_args}
                       {kw_args}
                       {pipe_ret_call}
                  """
        return textwrap.dedent(ret)


def assert_returns(t):
    def make(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            ret = f(*args, **kwargs)
            assert isinstance(ret, t), f
            return ret

        return wrapped

    return make


_types_not_parsed = set()


@assert_returns(Type)
def parse_type(t, builtin, context):
    # print(f"DD parse_type {t}")
    t = remove_ranges(t)
    t = re.sub(r'\s*\(\)\s*', '', t)

    param_name = context.param_name

    ret = None
    try:
        ret = builtin[t]
    except KeyError:
        pass

    if ret is not None:
        if isinstance(ret, Dict):
            if param_name in ['class_weight']:
                ret = DictIntToFloat()
        if isinstance(ret, Arr):
            if param_name in ['filenames']:
                ret = StringList()
        if isinstance(ret, (Arr)):
            if param_name in ['feature_names', 'target_names']:
                ret = List(String())
            if param_name in ['neigh_ind', 'is_inlier']:
                ret = Arr()
        return ret

    elts = parse_enum(t)
    if elts is not None:
        if all([is_string(elt) for elt in elts]):
            return Enum([StringValue(e.strip("\"'")) for e in elts])
        elts = [
            parse_type(elt, builtin, context.add(enum_elts=elts))
            for elt in elts
        ]
        # print("parsed enum elts:", elts)
        ret = Enum(elts)
        ret = simplify_enum(ret)
        return ret
    elif is_string(t):
        return StringValue(t.strip("'\""))
    elif is_int(t):
        return IntValue(t)
    else:
        if t and t not in _types_not_parsed:
            log.warning(f"failed to parse type: '{t}', context follows")
            log.warning(context)
            _types_not_parsed.add(t)
        return UnknownType(t)


def parse_types(doc, builtin, context, section='Parameters'):
    context = context.add(section=section)

    if doc is None:
        return {}
    elements = parse_params(doc, section)
    # print(f"DD elements: {elements}")

    ret = {}
    for element in elements:
        try:
            text = element.text.strip()
            if not text or text.startswith('..'):
                continue
            m = re.match(r'^\**(\w+)\s*:\s*([^\n]+)', text)
            if m is None:
                mm = re.match(r'^(\w+)\s*\n', text)
                if mm is not None:
                    ret[mm.group(1)] = UnknownType(text)
                    continue
                continue
            param_name = m.group(1)

            type_string = m.group(2)
            type_string = remove_default(type_string)
            type_string = remove_shape(type_string)
            type_string = type_string.strip(" \n\t,.;")

            if type_string.startswith('ref:'):
                continue

            if not type_string:
                log.warning(
                    f"no type string found for '{param_name}' in '{m.group(2)}', context follows"
                )
                log.warning(context)
                continue

            ty = parse_type(type_string,
                            builtin,
                            context=context.add(text=text,
                                                param_name=param_name,
                                                type_string=type_string))
            ret[param_name] = ty

        except Exception as e:
            log.error(f"error processing element: {element}")
            log.exception(e)
            raise

    if not ret and section == 'Returns':
        return parse_types(doc,
                           builtin,
                           context.add(return_fallback_to_yield=True),
                           section='Yields')

    if section == 'Yields':
        ret['__is_yield'] = True

    return ret


def parse_bunch_fetch(elements, builtin, context):
    context = context.add(group='bunch_fetch')
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
                ret[name] = parse_type(value,
                                       builtin,
                                       context=context.add(param_name=name,
                                                           line=line))
    ret = Bunch(ret)
    return ret


def parse_bunch_load(elements, context):
    text = ' '.join(e.text for e in elements)
    attributes = re.findall(r"'([^']+)'", text)
    # log.debug(attributes)
    types = dict(data=Arr(),
                 target=Arr(),
                 data_filename=String(),
                 target_filename=String(),
                 DESCR=String(),
                 filename=String(),
                 target_names=Arr(),
                 feature_names=List(String()),
                 images=Arr(),
                 filenames=Arr())
    return Bunch({k: types[k] for k in attributes})


def parse_bunch(doc, builtin, context, section='Returns'):
    elements = parse_params(doc, section=section)
    assert elements, (doc, parse(doc))

    if 'attributes are' in elements[0].text or (
            len(elements) > 1 and 'attributes are' in elements[1].text):
        return parse_bunch_load(elements, context)
    else:
        return parse_bunch_fetch(elements, builtin, context)


def dummy_ufunc(x, /, out=None, where=True):
    pass


ufunc_signature = inspect.signature(dummy_ufunc)


def doc_signature(f):
    """Attempt to parse the signature of a function at the beginning of
    its documentation. Useful for many numpy functions.

    """
    doc = inspect.getdoc(f)
    if not doc:
        # print(f"DD doc_signature: no doc for {qualname(f)}")
        return None
    m = re.search(re_parser.doc_sig, doc)
    if m is None:
        # if not doc.startswith("Not implemented"):
        #     doc_first = "\n".join(doc.split("\n")[:1])
        #     log.debug(
        #       f'doc_signature: no signature found for {qualname(f)} in doc {doc_first}...'
        #     )
        return None
    name = m.group(1)
    args = m.group(2)
    args = re.sub(r'\s+', ' ', args)

    replacements = {
        '(d0, d1, ..., dn)': '(d)',
        '(a1, a2, ...)': '(a)',
    }
    args = replacements.get(args, args)

    dummy_s = f"def __dummy_f_doc_sig{args}: pass"
    # globs = dict()
    locs = dict()
    # print(f"DD doc sig: exec {dummy_s}")
    orig_dummy = dummy_s
    dummy_s = re.sub(r'<no value>', 'None', dummy_s)
    dummy_s = re.sub(r'\[, start\[, end\]\]', ', start=None, end=None',
                     dummy_s)
    dummy_s = re.sub(r'\[x, y\]', 'x=None, y=None', dummy_s)
    dummy_s = re.sub(r',\],', r'],', dummy_s)
    dummy_s = re.sub(r'\[(\s*,?\s*)(\w+)(\s*,?\s*)\]', r'\1\2=None\3', dummy_s)
    dummy_s = re.sub(r'\(a1, a2, \.\.\.\)', 'a', dummy_s)
    dummy_s = re.sub(r'dtype=(np\.\w+)', 'dtype="\1"', dummy_s)

    # Reordering should not break the bindings, since we pass all
    # arguments as named.
    dummy_s = re.sub(r'start=None, stop,', 'stop, start=None,', dummy_s)
    try:
        exec(dummy_s, globals(), locs)
    except Exception:
        log.warning(
            f"doc_sig: could not parse synthetic sig for {name}: '{dummy_s}' ({orig_dummy})"
        )
        return None
    dummy = locs['__dummy_f_doc_sig']
    sig = inspect.signature(dummy)

    if hasattr(f, '__self__'):
        assert list(sig.parameters.values())[0].name != 'self'
    elif len(name.split('.')) > 1:
        # The doc is like a.cumsum(b, c): it is most probably an
        # instance method. Add self since the function does not have a
        # __self__ attribute.
        self_param = inspect.Parameter('self',
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
        new_params = [self_param] + list(sig.parameters.values())
        sig = sig.replace(parameters=new_params)
    # print(f"DD dummy doc sig function: {dummy}: {sig}")
    return sig


def signature(f):
    try:
        if hasattr(f, '_parse_args'):
            # Signature the scipy way.
            return inspect.signature(f._parse_args)
        else:
            return inspect.signature(f)
    except ValueError:
        import numpy
        if isinstance(f, numpy.ufunc):
            return ufunc_signature
        else:
            try:
                doc_sig = doc_signature(f)
            except Exception as e:
                log.warning(f"doc_signature() raised an exception: {e}")
                raise NoSignature(f)

            if doc_sig is not None:
                return doc_sig
            else:
                raise NoSignature(f)


class Function:
    def __init__(self,
                 name,
                 function,
                 overrides,
                 registry,
                 builtin,
                 namespace='__wrap_namespace'):
        self.name = name
        self.name.apply_overrides(overrides)

        registry.add(self.name, function)
        
        self.overrides = overrides
        self.function = function
        self.context = TypeContext(
            function=function, function_qualname=self.name.full_python_name())
        doc = inspect.getdoc(function)
        raw_doc = str(getattr(function, '__doc__', ''))
        sig = self._signature()
        fixed_values = overrides.fixed_values(self.name.full_python_name())
        parameters = self._build_parameters(sig, raw_doc, fixed_values,
                                            builtin, self.context)
        ret = self._build_ret(sig, raw_doc, fixed_values, builtin)
        self.wrapper = Wrapper(name, doc, parameters, ret, "function",
                               namespace)
        overrides.notice_function(self)

    def iter_types(self):
        for param in self.wrapper.parameters:
            yield param.ty
        yield self.wrapper.ret.ty

    # def _ml_name(self):
    #     return self.name.ml_name()
    #     # warning: all overrides are resolved based on the function
    #     # qualname, not python_name (which may in some rare cases be
    #     # different)
    #     ml = self.overrides.ml_name(self.name.full_python_name())
    #     if ml is not None:
    #         return ml
    #     return mlid(python_name)

    def _signature(self):
        sig = self.overrides.signature(self.name.full_python_name())
        if sig is not None:
            return sig
        return signature(self.function)

    def _build_parameters(self, sig, doc, fixed_values, builtin, context):
        context = context.add(group="params")
        param_types = self.overrides.param_types(self.name.full_python_name())
        if param_types is None:
            param_types = parse_types(doc,
                                      builtin,
                                      context=context,
                                      section='Parameters')
            # Make sure all params are in param_types, so that the
            # overrides trigger (even in case the params were not
            # found parsing the doc).
            for k in sig.parameters.keys():
                if k not in param_types:
                    param_types[k] = UnknownType(
                        f"{k} not found in parsed types {list(param_types.keys())}"
                    )
            param_types = self.overrides.types(self.name.full_python_name(),
                                               param_types, context)

        parameters = []
        # Some functions in Scipy have both an n and an N parameter,
        # which are both mapped to n in OCaml. Detect and fix it.
        seen_ml_names = set()
        for python_name, parameter in sig.parameters.items():
            ty = param_types[python_name]
            param = Parameter(LowerName(python_name, self.name), ty, parameter)
            while param.name.ml_name() in seen_ml_names:
                param.name.adjust()
            seen_ml_names.add(param.name.ml_name())
            if python_name in fixed_values:
                param.fixed_value = fixed_values[python_name][0]
            if param.is_star_star():
                if not isinstance(param.ty, UnknownType):
                    log.warning(
                        f"overriding {param} to have type (string * Py.Object.t) list"
                    )
                param.ty = StarStar()
            if param.is_star():
                if not isinstance(param.ty, List):
                    if not isinstance(param.ty, UnknownType):
                        log.warning(
                            f"overriding {param} to have type List(PyObject())"
                        )
                    param.ty = List(PyObject())

            parameters.append(param)

        self.overrides.parameters(parameters)

        return parameters

    def _build_ret(self, sig, doc, fixed_values, builtin):
        context = self.context.add(group="ret")

        if self.overrides.returns_bunch(self.name.full_python_name()):
            ret_type = parse_bunch(doc,
                                   builtin,
                                   context=context,
                                   section='Returns')
        else:
            ret_type_elements = parse_types(doc,
                                            builtin,
                                            context,
                                            section='Returns')

            for k, v in fixed_values.items():
                if v[1] and k in ret_type_elements:
                    del ret_type_elements[k]
            ret_type = make_return_type(ret_type_elements)

        over_ret_type = self.overrides.ret_type(
            self.name.full_python_name(), context.add(auto_type=ret_type))

        if over_ret_type is not None:
            ret_type = over_ret_type

        return Return(ret_type, self.name)

    def ml(self):
        return self.wrapper.ml()

    def mli(self, doc=True):
        return self.wrapper.mli(doc=doc)

    def md(self):
        return self.wrapper.md()

    def examples(self):
        f = io.StringIO()
        write_examples(f, self.function)
        return f.getvalue()

    def write_to_ml(self, f):
        f.write(self.ml())

    def write_to_mli(self, f):
        f.write(self.mli())

    def write_to_md(self, f):
        f.write(self.md())

    def write_examples_to(self, f):
        write_examples(f, self.function)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = f"{type(self).__name__} {self.wrapper.name.full_python_name()}{self._signature()}:\n"
        for param in self.wrapper.parameters:
            ret += f"    {param} ->\n"
        ret += f"    {self.wrapper.ret}\n"
        ret += f"  {self.mli(doc=False)}"
        # ret += f"  {self.ml('Sklearn')}"
        return ret


class DummyFunction:
    def __init__(self):
        pass

    def write_to_ml(self, *args, **kwargs):
        pass

    def write_to_mli(self, *args, **kwargs):
        pass

    def write_to_md(self, *args, **kwargs):
        pass

    def write_examples_to(self, *args, **kwargs):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Method(Function):
    def __init__(self, name, function, overrides, registry, builtin):
        super().__init__(name,
                         function,
                         overrides,
                         registry,
                         builtin,
                         namespace='(to_pyobject self)')
        self.wrapper.doc_type = "method"

    def _build_parameters(self, *args, **kwargs):
        super_params = super()._build_parameters(*args, **kwargs)
        if getattr(self.function, '__self__', None) is None and (
                not super_params
                or super_params[0].name.python_name() != 'self'):
            log.warning(
                f'ignoring method {self.name.full_python_name()}, arg 1 != self: {self._signature()}'
            )
            raise OutsideScope(self.function)
        params = [x for x in super_params if x.name.python_name() != 'self']
        params.append(SelfParameter())
        return params


class Ctor(Function):
    def __init__(self, name, function, overrides, registry, builtin):
        super().__init__(name,
                         function,
                         overrides,
                         registry,
                         builtin,
                         namespace='__wrap_namespace')
        self.wrapper.doc_type = "constructor and attributes"
        self.name.set_ml_name('create')

    def _build_ret(self, _signature, _doc, _fixed_values, _builtin):
        return Return(Self(), self.name)

    # def _ml_name(self, _python_name):
    #     return 'create'


def re_compile(regex):
    if isinstance(regex, type):
        return regex
    try:
        return re.compile(regex)
    except re.error as e:
        log.error(f"error compiling regex {regex}: {e}")
        raise


def compile_dict_keys(dic):
    return {re_compile(k): v for k, v in dic.items()}


class Overrides:
    def __init__(self, overrides, scope=r''):
        self.overrides = self._compile_regexes(overrides)
        self.triggered = set()
        self.scope = re.compile(scope)
        self._on_function = []
        self._blacklisted_submodules = []
        self._filters = []
        
    def is_blacklisted_submodule(self, m):
        for x in self._blacklisted_submodules:
            if x is m:
                return True
        return False

    def add_blacklisted_submodule(self, m):
        self._blacklisted_submodules.append(m)

    def on_function(self, f):
        self._on_function.append(f)

    def notice_function(self, f):
        # print(f"DD noticing function {f}, calling callbacks")
        for cb in self._on_function:
            try:
                cb(f)
            except Exception as e:
                log.warning(
                    f"{f}: on_function callback raised an exception: {e}")
                raise

    def add_filter(self, f):
        self._filters.append(f)
            
    def check_scope(self, full_python_name, element):
        if re.match(self.scope, full_python_name) is None:
            return False
        for f in self._filters:
            if not f(full_python_name, element):
                return False
        # print(f"DD check scope {full_python_name} / {self.scope.pattern}: {ret}")
        return True

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

    def parameters(self, parameters):
        for param in parameters:
            for dic in self._iter_function_matches(
                    param.name.full_python_name(), 'parameters'):
                if not dic.get('name', True):
                    param.remove_name()

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

    def ret_type(self, qualname, context):
        ret = []
        for dic in self._iter_function_matches(qualname, 'ret_type'):
            if 'ret_type' in dic:
                ret.append(dic['ret_type'])
        assert len(ret) <= 1, ("several ret_type found for function", qualname,
                               ret)
        if ret:
            return ret[0](context)
        return None

    def types(self, qualname, param_types, context):
        param_types = dict(param_types)
        for dic in self._iter_function_matches(qualname, 'types'):
            # use list() to freeze the list, we will be modifying param_types
            for k in list(param_types.keys()):
                for param_re, ty in dic.get('types', {}).items():
                    if re.search(param_re, k) is not None:
                        auto_type = param_types[k]
                        param_types[k] = ty(context.add(auto_type=auto_type))
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
            if hasattr(k, 'pattern'):
                k = k.pattern
            if k not in self.triggered:
                not_triggered.add(k)
        if not_triggered:
            log.warning(f"overrides: keys not triggered: {not_triggered}")


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


def map_enum(t, f):
    if isinstance(t, Enum):
        return type(t)([f(x) for x in t.elements])
    elif isinstance(t, Optional):
        return type(t)(map_enum(t.t, f))
    else:
        return t


def scoring_param(context):
    import sklearn.metrics
    # log.debug(list(sklearn.metrics.SCORERS.keys()))
    scoring = Enum([StringValue(x) for x in sklearn.metrics.SCORERS.keys()])
    scoring.tag_name = lambda: "Score"

    # log.debug(f"scoring context:\n{context}")

    def to_scoring(t):
        if isinstance(t, String):
            return scoring
        elif isinstance(t, UnknownType) and t.text in [
                'tuple', 'list/tuple', 'tuple/list'
        ]:
            return List(scoring)
        else:
            return t

    return map_enum(context.auto_type, to_scoring)


def if_unknown(t):
    def f(context):
        if isinstance(context.auto_type, (PyObject, UnknownType)):
            return t
        else:
            return context.auto_type

    return f


sklearn_overrides = {
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
    r'__iter__$':
    dict(ml_name='iter', ret_type=Generator(Dict())),
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
    dict(ret_type=Tuple([ArrPyList(), ArrPyList()])),
    r'^NearestCentroid\.fit$':
    dict(param_types=dict(X=Arr(), y=Arr())),
    r'MultiLabelBinarizer\.(fit_transform|fit)':
    dict(types={'^y$': ArrPyList()}),
    r'\.(decision_function|predict|predict_proba|fit_predict|transform|fit_transform)$':
    dict(ret_type=Arr(), types={r'^X$': Arr()}),
    r'\.(fit|partial_fit)$':
    dict(ret_type=Self()),
    r'Pipeline$':
    dict(param_types=dict(steps=estimator_alist,
                          memory=Enum([NoneValue(
                          ), String(), UnknownType('Joblib Memory')]),
                          verbose=Bool())),
    r'load_iris$':
    dict(ret_type=Bunch({
        'data': Arr(),
        'target': Arr(),
        'target_names': Arr(),
        'feature_names': List(String()),
        'DESCR': String(),
        'filename': String()
    })),
    r'load_boston$':  # feature_names is missing from docs
    dict(ret_type=Bunch({
        'data': Arr(),
        'target': Arr(),
        'feature_names': List(String()),
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
            r'^(base_)?estimator$': Estimator(),
            r'scoring': scoring_param,
            r'labels_': if_unknown(Arr()),
            r'^decision_tree$': if_unknown(BaseType('sklearn.tree.BaseDecisionTree')),
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
    r'\.auc$':
    dict(param_types=dict(x=Arr(), y=Arr()), ret_type=Float()),
    r'\.classification_report$':
    dict(ret_type=Enum([String(), ClassificationReport()])),
    # hamming_loss() is documented as returning int or float, but I
    # don't see how it can ever return an int.
    r'\.hamming_loss$':
    dict(ret_type=Float()),
    r'(CV|ParameterGrid)$':
    dict(types={
        r'^param_grid$': Enum([ParamGridDict(),
                               List(ParamGridDict())])
    }),
    r'(ParameterSampler|RandomizedSearchCV)$':
    dict(types={
        r'^param_distributions$': Enum([ParamDistributionsDict(),
                                        List(ParamDistributionsDict())])
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
        }),
    r'\.get_n_splits$': dict(ret_type=Int()),
    r'\.make_pipeline$': dict(types={r'^steps$': List(Estimator())}),
    r'Birch$': dict(types={r'^n_clusters$': Enum([NoneValue(), Int(), ClusterEstimator()])}),
    r'^SpectralBiclustering$': dict(types={r'^n_clusters$': Enum([Int(), Tuple([Int(), Int()])])}),
    r'export_graphviz$': dict(ret_type=Optional(String())),
    r'SimpleImputer$': dict(types={r'^strategy$': Enum([StringValue('mean'), StringValue('median'),
                                                        StringValue('most_frequent'), StringValue('constant')])}),
    r'ColumnTransformer$': dict(types={
        r'^transformers$': List(Tuple([String(), Transformer(),
                                       Enum([String(), Int(), List(String()), List(Int()),
                                             Slice(), Arr(), Callable()])]))}),
    r'TransformedTargetRegressor$': dict(types={r'^regressor$': Regressor(),
                                                r'^transformer$': Transformer()}),
    r'\.make_column_transformer$': dict(types={r'^transformers$':
                                               List(Tuple([Transformer(),
                                                           Enum([String(), Int(), List(String()), List(Int()),
                                                                 Slice(), Arr(), Callable()])]))},
                                        ret_type=WrappedModule('Sklearn.Compose.ColumnTransformer'))
}

scipy_overrides = {
    r'': dict(types={r'^loc$': Float(), '^scale$': Float()}),
    r'mquantiles$':
    dict(param_types=dict(a=Arr(),
                          prob=Arr(),
                          alphap=Float(),
                          betap=Float(),
                          axis=Int(),
                          limit=Tuple([Float(), Float()])),
         ret_type=Arr()),
    r'todense$':
    dict(ret_type=Arr())
}


def zeros_sig():
    import numpy as np
    sig = inspect.signature(np.ones)
    params = sig.parameters.copy()
    params['shape'] = params['shape'].replace(
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    params.move_to_end('shape', last=False)
    sig = sig.replace(parameters=params.values())
    return sig


def fix_ufunc_ret(context):
    import numpy as np
    if isinstance(context.function, np.ufunc) and isinstance(
            context.auto_type, (PyObject, UnknownType)):
        return Ndarray(inside_np=True)
    else:
        return context.auto_type


def dummy_seed(seed):
    pass


seed_sig = inspect.signature(dummy_seed)


def dummy_randn(*d):
    pass


randn_sig = inspect.signature(dummy_randn)

np_overrides = {
    r'':
    dict(types={r'^(?:shape|newshape)$': List(Int)}),
    r'^numpy.[^.]+$':
    dict(ret_type=fix_ufunc_ret),
    r'\.(?:a|shape|x)$':
    dict(name=False),
    r'^(?:ones|zeros)$':
    dict(signature=zeros_sig()),
    r'ndarray\.__getitem__$':
    dict(ml_name='get',
         types={
             r'key': WrappedModule('Wrap_utils.Index'),
             'value': Self()
         },
         ret_type=Self()),
    r'ndarray\.__setitem__$':
    dict(ml_name='set',
         types={
             r'key': WrappedModule('Wrap_utils.Index'),
             'value': Self()
         },
         ret_type=Unit()),
    r'^numpy\.random\.seed$':
    dict(signature=seed_sig, param_types=dict(seed=Int()), ret_type=Unit()),
    r'^numpy\.random\.randn$':
    dict(signature=randn_sig,
         param_types=dict(d=List(Int())),
         ret_type=Ndarray(inside_np=True)),
    r'^numpy\.random\.random_sample$':
    dict(ret_type=Ndarray(inside_np=True)),
    r'^numpy\.random\.':
    dict(types={r'^size$': List(Int())}),
}


def write_version(package, build_dir, registry):
    full_version = distutils.version.LooseVersion(
        package.__version__).version
    full_version_ml = '[' + '; '.join([f'"{x}"' for x in full_version]) + ']'
    version_ml = '(' + ', '.join(str(x) for x in full_version[:2]) + ')'
    ml = build_dir / 'wrap_version.ml'
    with open(ml, 'w') as f:
        registry.add_generated_file(ml)
        f.write(f'let full_version = {full_version_ml}\n')
        f.write(f'let version = {version_ml}\n')


def scipy_on_function(f):
    import scipy.stats.distributions
    # print(f"DD calling scipy callback on {f.qualname}")
    if f.name.full_python_name().startswith('scipy.stats'):
        gen = f'{f.name.python_name()}_gen'
        # print(f"DD testing for hasattr {gen}")
        if hasattr(scipy.stats.distributions, gen):
            # f.wrapper.ret.ty = WrappedModule(
            #     f'Scipy.Stats.Distributions.{make_module_name(gen)}')
            f.wrapper.ret.ty = BaseType(f'scipy.stats.distributions.{gen}')
            # print(f"DD adjusted return type on {f}")


def sklearn_pkg():
    import sklearn
    over = Overrides(sklearn_overrides, r'^sklearn(\..+)?')
    registry = Registry()
    builtin = sklearn_builtin_types
    pkg = Package(sklearn, over, registry, builtin)
    return pkg


def scipy_filter(full_python_name, element):
    import scipy
    # Oh scipy...
    # Most of these have key.__name__ == value, but not all.
    once = {
        scipy.fftpack.convolve: 'scipy.fftpack.convolve',
        scipy.integrate.lsoda: 'scipy.integrate.lsoda',
        scipy.integrate.odepack: 'scipy.integrate.odepack',
        scipy.integrate.quadpack: 'scipy.integrate.quadpack',
        scipy.integrate.vode: 'scipy.integrate.vode',
        scipy.integrate: 'scipy.integrate',
        scipy.interpolate.dfitpack: 'scipy.interpolate.dfitpack',
        scipy.interpolate.fitpack: 'scipy.interpolate.fitpack',
        scipy.interpolate: 'scipy.interpolate',
        scipy.io.byteordercodes: 'scipy.io.byteordercodes',
        scipy.io.matlab.mio5_params: 'scipy.io.matlab.mio5_params',
        scipy.io.matlab.miobase: 'scipy.io.matlab.miobase',
        scipy.linalg.decomp: 'scipy.linalg.decomp',
        scipy.linalg.decomp_svd: 'scipy.linalg.decomp_svd',
        scipy.linalg: "scipy.linalg",
        scipy.ndimage.filters: 'scipy.ndimage.filters',
        scipy.ndimage.measurements: 'scipy.ndimage.measurements',
        scipy.ndimage.morphology: 'scipy.ndimage.morphology',
        scipy.optimize.minpack2: 'scipy.optimize.minpack2',
        scipy.optimize.moduleTNC: 'scipy.optimize.moduleTNC',
        scipy.optimize: "scipy.optimize",
        scipy.signal.signaltools: 'scipy.signal.signaltools',
        scipy.signal.sigtools: 'scipy.signal.sigtools',
        scipy.spatial.distance: 'scipy.spatial.distance',
        scipy.spatial.qhull: 'scipy.spatial.qhull',
        scipy.special.specfun: "scipy.special.specfun",
        scipy.special: "scipy.special",
        scipy.stats.distributions: "scipy.stats.distributions",
        scipy.stats.mstats_basic: 'scipy.stats.mstats_basic',
        scipy.stats.mvn: 'scipy.stats.mvn',
        scipy.stats.statlib: 'scipy.stats.statlib',
        scipy.stats.stats: 'scipy.stats.stats',
    }
    wanted_name = once.get(element, None)
    if wanted_name is not None:
        return wanted_name == full_python_name
    if element in [scipy._lib.doccer]:
        return False
    return True


def scipy_pkg():
    builtin = scipy_builtin_types
    import scipy
    over = Overrides(scipy_overrides, r'^scipy(\..+)?')
    over.add_filter(scipy_filter)
    over.on_function(scipy_on_function)
    registry = Registry()
    pkg = Package(scipy, over, registry, builtin)
    return pkg

def numpy_filter(full_python_name, element):
    import numpy
    # Most of these have key.__name__ == value, but not all.
    once = {
        numpy.fft: 'numpy.fft',
        numpy.fft.helper: 'numpy.fft.helper',
        numpy.polynomial.polyutils: 'numpy.polynomial.polyutils',
        numpy.linalg: 'numpy.linalg',
        numpy.linalg.linalg: 'numpy.linalg.linalg',
        numpy.linalg.lapack_lite: 'numpy.linalg.lapack_lite',
    }
    wanted_name = once.get(element, None)
    if wanted_name is not None:
        return wanted_name == full_python_name
    if element.__name__.startswith('numpy.core'):
        return False
    return True


def numpy_pkg():
    builtin = numpy_builtin_types
    import numpy

    over = Overrides(
        np_overrides,
        r'^(?!.*\bchararray\b.*)(?!numpy\.dtype)numpy(\..+)?')
    over.add_filter(numpy_filter)
    registry = Registry()
    # pkg = Module(in_ml_namespace('Np', UpperName('numpy', None)), numpy, over,
    #              registry, builtin)
    pkg = Module(UpperName('numpy', parent=None, ml_name='NumpyRaw'),
                 numpy, over, registry, builtin)
    return pkg


def numpy_fun(fn):
    builtin = numpy_builtin_types
    import numpy

    over = Overrides(
        np_overrides,
        r'^(?!.*\bchararray\b.*)(?!numpy\.dtype)numpy(\..+)?')
    over.add_filter(numpy_filter)
    registry = Registry()
    # pkg = Module(in_ml_namespace('Np', UpperName('numpy', None)), numpy, over,
    #              registry, builtin)
    name = LowerName(fn.__name__, None, mlid(fn.__name__))
    return Function(name, fn, over, registry, builtin)


def sklearn_fun(fn):
    import sklearn
    over = Overrides(sklearn_overrides, r'^sklearn(\..+)?')
    registry = Registry()
    builtin = sklearn_builtin_types
    name = LowerName(fn.__name__, None, mlid(fn.__name__))
    return Function(name, fn, over, registry, builtin)


def sklearn_method(fn):
    import sklearn
    over = Overrides(sklearn_overrides, r'^sklearn(\..+)?')
    registry = Registry()
    builtin = sklearn_builtin_types
    name = LowerName(fn.__name__, None, mlid(fn.__name__))
    return Method(name, fn, over, registry, builtin)


def write(pkg, path):
    pkg.write(path)
    try:
        py = pkg.pkg
    except AttributeError:
        py = pkg.module
    write_version(py, path, pkg.registry)
    pkg.registry.write(path)
    pkg.overrides.report_not_triggered()
    pkg.registry.report_generated()


def main():
    # There are FutureWarnings about deprecated items. We want to
    # catch them in order not to wrap them.
    import warnings
    warnings.simplefilter('error', FutureWarning)
    warnings.simplefilter('error', DeprecationWarning)

    build_dir = pathlib.Path('.')
    # build_dir.mkdir(parents=True, exist_ok=True)

    import sys
    if len(sys.argv) <= 1:
        log.warning("skdoc.py: no argument passed, not doing anything")
        return

    pkg_maker = dict(sklearn=sklearn_pkg, scipy=scipy_pkg, numpy=numpy_pkg)

    mode = sys.argv[1]
    pkg_name = sys.argv[2]

    assert pkg_name in pkg_maker, (f"unknown package {pkg_name}, available: {sorted(pkg_maker.keys())}")
    pkg = pkg_maker[pkg_name]()

    if mode == "build":
        write(pkg, build_dir)

    elif mode == "doc":
        doc_dir = pathlib.Path(sys.argv[3])
        pkg.write_doc(doc_dir)

    elif mode == "examples":
        pkg = sklearn_pkg()
        path = pathlib.Path('./_build/examples/auto/')
        log.info(f"extracting examples to {path}")
        pkg.write_examples(path)

    else:
        log.error(f"unknown mode {mode}, available: build, doc, examples")


assert test_remove_shape(), "some remove_shape() tests failed"
assert test_parse_enum(), "some parse_enum() tests failed"

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='skdoc.log')
    main()
