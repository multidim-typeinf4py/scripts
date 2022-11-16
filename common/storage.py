from __future__ import annotations

import enum
import pathlib

from libcst.codemod.visitors._apply_type_annotations import Annotations
import libcst as cst

import pandas as pd

import pandera as pa
import pandera.typing as pt


class Category(enum.Enum):
    VARIABLE = "variable"
    CALLABLE_RETURN = "function"
    CALLABLE_PARAMETER = "parameter"
    CLASS_ATTR = "classdef"

    def __str__(self) -> str:
        return self.name


class Schema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=Category)
    qname: pt.Series[str] = pa.Field()
    anno: pt.Series[str] = pa.Field(nullable=True, coerce=True)


SchemaColumns = list(Schema.to_schema().columns.keys())


def _stringify(node: cst.CSTNode | None) -> str | None:
    match node:
        case cst.Annotation(cst.Name(name)):
            return name
        case cst.Annotation(cst.Module()):
            m: cst.Module = node.annotation
            return m.code
        case cst.Name(name):
            return name
        case None:
            return None
        case _:
            raise AssertionError(f"Unhandled node: {node}")


class TypeCollection:
    @pa.check_types
    def __init__(self, df: pt.DataFrame[Schema]) -> None:
        self._df = df

    @staticmethod
    def empty() -> TypeCollection:
        return TypeCollection(
            df=pd.DataFrame(columns=SchemaColumns).pipe(pt.DataFrame[Schema])
        )

    @staticmethod
    def from_annotations(
        file: pathlib.Path, annotations: Annotations
    ) -> TypeCollection:

        from pandas._libs import missing

        filename = str(file)

        contents = list()
        for fkey, fanno in annotations.functions.items():
            # NOTE: if fanno.returns is None, this is accurate!, as:
            # NOTE: Functions without a return type are assumed to return None
            contents.append(
                (
                    filename,
                    Category.CALLABLE_RETURN,
                    fkey.name,
                    _stringify(fanno.returns) or "None",
                )
            )

            # NOTE: if param.annotation is None, this is NOT accurate!, as:
            # NOTE: unlike functions, no assumption of None is given
            # NOTE: therefore, we must differentiate between "None" and None, and mark
            # NOTE: the latter as INVALID!
            for param in fanno.parameters.params:
                contents.append(
                    (
                        filename,
                        Category.CALLABLE_PARAMETER,
                        f"{fkey.name}.{param.name.value}",
                        _stringify(param.annotation) or missing.NA,
                    )
                )

        for qname, anno in annotations.attributes.items():
            # NOTE: assignments to variables without an annotation are deemed INVALID
            contents.append(
                (filename, Category.VARIABLE, qname, _stringify(anno) or missing.NA)
            )

        for cqname, cdef in annotations.class_definitions.items():
            for stmt in cdef.body.body:
                # NOTE: No need to check for validity of `annassign.annotation`
                # NOTE: as cst.AnnAssign exists precisely so that the Annotation exists
                annassign: cst.AnnAssign = stmt.body[0]
                contents.append(
                    (
                        filename,
                        Category.CLASS_ATTR,
                        f"{cqname}.{_stringify(annassign.target)}",
                        _stringify(annassign.annotation),
                    )
                )

        df = pd.DataFrame(contents, columns=SchemaColumns)
        return TypeCollection(df.pipe(pt.DataFrame[Schema]))

    @pa.check_types
    def update(self, other: pt.DataFrame[Schema]) -> None:
        self._df = (
            pd.concat([self._df, other], ignore_index=True)
            .drop_duplicates(keep="last")
            .pipe(pt.DataFrame[Schema])
        )

    def merge(self, other: TypeCollection) -> None:
        self.update(other._df)
