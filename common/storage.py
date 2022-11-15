from __future__ import annotations

import enum
import functools
import pathlib

from libcst.codemod.visitors._apply_type_annotations import Annotations
import libcst as cst

import pandas as pd

import pandera as pa
import pandera.typing as pt


class Category(enum.Enum):
    ATTRIBUTE = "attribute"
    CALLABLE_RETURN = "function"
    CALLABLE_PARAMETER = "parameter"
    CLASS_DEF = "classdef"

    def __str__(self) -> str:
        return self.name


class Schema(pa.SchemaModel):
    file: pt.Series[str] = pa.Field()
    category: pt.Series[str] = pa.Field(isin=Category)
    qname: pt.Series[str] = pa.Field()
    anno: pt.Series[str] = pa.Field(nullable=True)


SchemaColumns = list(Schema.to_schema().columns.keys())


def _stringify(node: cst.CSTNode | None) -> str | None:
    if node is None:
        return None

    match node:
        case cst.Annotation(cst.Name(name)):
            return name
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
        filename = str(file)

        contents = list()
        for fkey, fanno in annotations.functions.items():
            contents.append(
                (
                    filename,
                    Category.CALLABLE_RETURN,
                    fkey.name,
                    _stringify(fanno.returns),
                )
            )

            for param in fanno.parameters.params:
                contents.append(
                    (
                        filename,
                        Category.CALLABLE_PARAMETER,
                        f"{fkey.name}.{param.name.value}",
                        _stringify(param.annotation),
                    )
                )

        for qname, anno in annotations.attributes.items():
            contents.append((filename, Category.ATTRIBUTE, qname, _stringify(anno)))

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
