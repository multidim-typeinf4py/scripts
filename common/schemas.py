from __future__ import annotations

import enum

import pandera as pa
import pandera.typing as pt


class TypeCollectionCategory(enum.Enum):
    VARIABLE = "variable",
    CALLABLE_RETURN = "function",
    CALLABLE_PARAMETER = "parameter",
    CLASS_ATTR = "classdef",

    def __str__(self) -> str:
        return self.name


class TypeCollectionSchema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=TypeCollectionCategory)
    qname: pt.Series[str] = pa.Field()
    anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)


TypeCollectionSchemaColumns = list(TypeCollectionSchema.to_schema().columns.keys())


class MergedAnnotationSchema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=TypeCollectionCategory)
    qname: pt.Series[str] = pa.Field()

    # TODO: anno will be prefixed by the repositories' respective naming
    # TODO: can this be schematised?


MergedAnnotationSchemaColumns = list(MergedAnnotationSchema.to_schema().columns.keys())
