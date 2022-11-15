from __future__ import annotations

import enum
import pathlib

from libcst.codemod.visitors._apply_type_annotations import Annotations
import libcst as cst
import libcst.codemod as codemod

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
    category: pt.Series[pa.Category] = pa.Field(
        isin={category.value for category in Category}
    )
    qname: pt.Series[str] = pa.Field()
    anno: pt.Series[str] = pa.Field(nullable=True)


def _stringify(node: cst.CSTNode | None) -> str | None:
    if node is None:
        return None
    return cst.Module([]).code_for_node(node)


class TypeCollection:
    @pa.check_types
    def __init__(self, df: pt.DataFrame[Schema]) -> None:
        self._df = df

    @staticmethod
    def empty() -> TypeCollection:
        return TypeCollection(df=pd.DataFrame().pipe(pt.DataFrame[Schema]))

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

        df = pd.DataFrame(contents, columns=list(Schema.__fields__.keys()))
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
