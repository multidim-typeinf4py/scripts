from __future__ import annotations

import enum

import pandera as pa
import pandera.typing as pt


class TypeCollectionCategory(enum.Enum):
    VARIABLE = "variable"
    CALLABLE_RETURN = "function"
    CALLABLE_PARAMETER = "parameter"
    CLASS_ATTR = "classdef"

    def __str__(self) -> str:
        return self.name


class SymbolSchema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=TypeCollectionCategory)
    qname: pt.Series[str] = pa.Field()
    qname_ssa: pt.Series[str] = pa.Field()


class TypeCollectionSchema(SymbolSchema):
    anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)


TypeCollectionSchemaColumns = list(TypeCollectionSchema.to_schema().columns.keys())


class InferredSchema(TypeCollectionSchema):
    method: pt.Series[str] = pa.Field()
    topn: pt.Series[int] = pa.Field(ge=0)


InferredSchemaColumns = list(InferredSchema.to_schema().columns.keys())


class MergedAnnotationSchema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=TypeCollectionCategory)
    qname: pt.Series[str] = pa.Field()

    # TODO: anno will be prefixed by the repositories' respective naming
    # TODO: can this be schematised?


MergedAnnotationSchemaColumns = list(MergedAnnotationSchema.to_schema().columns.keys())


class ContextCategory(enum.IntEnum):
    CALLABLE_RETURN = enum.auto()
    CALLABLE_PARAMETER = enum.auto()
    VARIABLE = enum.auto()
    INSTANCE_ATTR = enum.auto()
    CLASS_ATTR = enum.auto()

    def __str__(self) -> str:
        return self.name


class ContextSymbolSchema(TypeCollectionSchema):
    loop: pt.Series[int] = pa.Field()
    reassigned: pt.Series[int] = pa.Field()
    nested: pt.Series[int] = pa.Field()
    user_defined: pt.Series[int] = pa.Field()
    branching: pt.Series[int] = pa.Field()
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

ContextSymbolSchemaColumns = list(ContextSymbolSchema.to_schema().columns.keys())


