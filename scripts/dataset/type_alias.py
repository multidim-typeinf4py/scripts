import libcst
from libcst import matchers as m

TYPING_TYPE = (
    # typing.Type
    m.Attribute(m.Name("typing"), m.Name("Type"))
    | m.Name("Type")
    # typing.Type[...]
    | m.Subscript(value=m.Attribute(m.Name("typing"), m.Name("Type")))
    | m.Subscript(m.Name("Type"))
    # typing.TypeAlias (there is no subscript form)
    | m.Attribute(m.Name("typing"), m.Name("TypeAlias"))
    | m.Name("TypeAlias")
    # typing.TypeVar (there is no subscript form)
    | m.Attribute(m.Name("typing"), m.Name("TypeVar"))
    | m.Name("TypeVar")
    # typing.NewType
    | m.Attribute(m.Name("typing"), m.Name("NewType"))
    | m.Name("NewType")
)

BUILTIN_TYPE = (
    # builtins.type
    m.Attribute(m.Name("builtins"), m.Name("type"))
    | m.Name("type")
    # builtins.type[...]
    | m.Subscript(value=m.Attribute(m.Name("builtins"), m.Name("type")))
    | m.Subscript(m.Name("type"))
)


def is_type_alias(annotation: str | None) -> bool:
    if annotation is None:
        return False

    anno_as_expr = libcst.parse_expression(source=annotation)
    return m.matches(anno_as_expr, TYPING_TYPE | BUILTIN_TYPE)
