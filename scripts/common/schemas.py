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


class DatasetSchema(TypeCollectionSchema):
    rewritten: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    parametric: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    type_neutral: pt.Series[str] = pa.Field(nullable=True, coerce=True)
    is_type_alias: pt.Series[bool] = pa.Field()


class InferredSchema(TypeCollectionSchema):
    method: pt.Series[str] = pa.Field()
    topn: pt.Series[int] = pa.Field(ge=1)


class ContextCategory(enum.IntEnum):
    CALLABLE_RETURN = enum.auto()
    CALLABLE_PARAMETER = enum.auto()
    VARIABLE = enum.auto()
    INSTANCE_ATTR = enum.auto()

    def __str__(self) -> str:
        return self.name


class ScopePossibility(enum.IntFlag):
    IMPORTED = enum.auto()
    LOCAL = enum.auto()
    BUILTIN = enum.auto()

    BUILTIN_LOCAL = BUILTIN | LOCAL
    BUILTIN_IMPORTED = BUILTIN | IMPORTED
    LOCAL_IMPORTED = LOCAL | IMPORTED

    BUILTIN_LOCAL_IMPORTED = BUILTIN | LOCAL | IMPORTED

    def __str__(self) -> str:
        return self.name()

    @staticmethod
    def from_analysis(builtin: bool, local: bool, imported: bool) -> ScopePossibility:
        return ScopePossibility((int(builtin) << 2) | (int(local) << 1) | (int(imported) << 0))


class ContextSymbolSchema(TypeCollectionSchema):
    simple_name: pt.Series[str] = pa.Field()
    loop: pt.Series[int] = pa.Field()
    reassigned: pt.Series[int] = pa.Field()
    nested: pt.Series[int] = pa.Field()
    branching: pt.Series[int] = pa.Field()
    scope_analysis: pt.Series[int] = pa.Field(isin=ScopePossibility)
    ctxt_category: pt.Series[int] = pa.Field(isin=ContextCategory)


class ContextDatasetSchema(pa.SchemaModel):
    method: pt.Series[str] = pa.Field()
    file: pt.Series[str] = pa.Field()
    qname_ssa: pt.Series[str] = pa.Field()
    anno_gt: pt.Series[str] = pa.Field(nullable=True)
    anno_ta: pt.Series[str] = pa.Field(nullable=True)
    score: pt.Series[int] = pa.Field(ge=-1.0, le=1.0)

    loop: pt.Series[int] = pa.Field()
    reassigned: pt.Series[int] = pa.Field()
    nested: pt.Series[int] = pa.Field()
    user_defined: pt.Series[int] = pa.Field()
    ctxt_category: pt.Series[int] = pa.Field(isin=ContextCategory)
