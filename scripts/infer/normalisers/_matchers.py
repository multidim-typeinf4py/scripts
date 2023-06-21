import libcst
from libcst import matchers as m

### Matchers ###


UNION_ = m.Name("Union") | m.Attribute(m.Name(), m.Name("Union"))
OPTIONAL_ = m.Name("Optional") | m.Attribute(m.Name(), m.Name("Optional"))

_QUALIFIED_BOOL_LIT = m.Attribute(m.Name("builtins"), m.Name("False") | m.Name("True"))
_UNQUALIFIED_BOOL_LIT = m.Name("False") | m.Name("True")

_QUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(
    m.Attribute(m.Name(), m.Name("Literal"))
)
_UNQUALIFIED_SUBSCRIPT_LITERAL = m.Subscript(m.Name("Literal"))

ANY_ = m.Attribute(m.Name(), m.Name("Any")) | m.Name("Any")

DICT_NAME = m.Name("Dict")
DICT_ATTR_ = m.Attribute(m.Name(), m.Name("Dict"))

TUPLE_ATTR_ = m.Attribute(m.Name(), m.Name("Tuple"))
TUPLE_NAME_ = m.Name("Tuple")

LIST_ATTR_ = m.Attribute(m.Name(), m.Name("List"))
LIST_NAME_ = m.Name("List")

SET_ATTR_ = m.Attribute(m.Name(), m.Name("Set"))
SET_NAME_ = m.Name("Set")

TEXT_ATTR_ = m.Attribute(m.Name(), m.Name("Text"))
TEXT_NAME_ = m.Name("Text")

OPTIONAL_ATTR_ = m.Attribute(m.Name(), m.Name("Optional"))
OPTIONAL_NAME_ = m.Name("Optional")

FINAL_ATTR_ = m.Attribute(m.Name(), m.Name("Final"))
FINAL_NAME_ = m.Name("Final")


### Replacements

TUPLE_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("Tuple"))
LIST_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("List"))
DICT_ = libcst.Attribute(libcst.Name("typing"), libcst.Name("Dict"))

