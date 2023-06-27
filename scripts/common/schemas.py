from __future__ import annotations

import enum

import pandera as pa
import pandera.typing as pt


class TypeCollectionCategory(enum.Enum):
    VARIABLE = "variable"
    CALLABLE_RETURN = "function"
    CALLABLE_PARAMETER = "parameter"

    def __str__(self) -> str:
        return self.name

class SymbolSchema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=TypeCollectionCategory)
    qname: pt.Series[str] = pa.Field()
    qname_ssa: pt.Series[str] = pa.Field()


class TypeCollectionSchema(SymbolSchema):
    anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)


class ExtendedTypeCollectionSchema(SymbolSchema):
    raw_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    depth_limited_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    adjusted_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    base_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)




class InferredSchema(TypeCollectionSchema):
    method: pt.Series[str] = pa.Field()
    topn: pt.Series[int] = pa.Field(ge=1)


class ExtendedInferredSchema(InferredSchema):
    # is_type_alias: pt.Series[bool] = pa.Field()
    parametric_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    # type_neutral_anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    # common_or_rare: pt.Series[str] = pa.Field(nullable=True, isin=["common", "rare"])
    simple_or_complex: pt.Series[str] = pa.Field(nullable=True, isin=["simple", "complex"])


class ContextCategory(enum.IntEnum):
    # -> f() -> ...
    CALLABLE_RETURN = enum.auto()

    # -> f(a: ...)
    CALLABLE_PARAMETER = enum.auto()

    # -> a = 10
    SINGLE_TARGET_ASSIGN = enum.auto()

    # -> a: int = 10
    ANN_ASSIGN = enum.auto()

    # -> a += 5
    AUG_ASSIGN = enum.auto()

    # a, b = 10 or a = b = 10
    MULTI_TARGET_ASSIGN = enum.auto()

    # a: int in a class
    INSTANCE_ATTRIBUTE = enum.auto()

    # for x in y:
    FOR_TARGET = enum.auto()

    # with open as f
    WITH_TARGET = enum.auto()

    def __str__(self) -> str:
        return self.name


class ContextSymbolSchema(SymbolSchema):
    # Read above to recognise the category
    context_category: pt.Series[int] = pa.Field(isin=ContextCategory)

    # Does the annotatable occur in a loop
    loop: pt.Series[int] = pa.Field()

    # Is this annotatable used on the LHS of multiple assignments
    reassigned: pt.Series[int] = pa.Field()

    # Is the annotatable inside a class inside a class, or a function inside a function
    nested: pt.Series[int] = pa.Field()

    # Is the annotatable inside some form of flow control (try, except(*), if, else)
    flow_control: pt.Series[int] = pa.Field()

    # Does the annotation attached require being able to follow an import?
    import_source: pt.Series[int] = pa.Field()

    # Does the annotation attached require knowing the builtin scope
    builtin_source: pt.Series[int] = pa.Field()

    # Does the annotation attached require knowing the local scope
    local_source: pt.Series[int] = pa.Field()


