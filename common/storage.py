from __future__ import annotations

import functools
import pathlib

from libcst.codemod.visitors._apply_type_annotations import Annotations
import libcst as cst

import pandas as pd
import pandera as pa
import pandera.typing as pt


from ._helper import _stringify
from .schemas import (
    TypeCollectionCategory,
    TypeCollectionSchema,
    TypeCollectionSchemaColumns,
)

from .schemas import MergedAnnotationSchema, MergedAnnotationSchemaColumns


class TypeCollection:
    @pa.check_types
    def __init__(self, df: pt.DataFrame[TypeCollectionSchema]) -> None:
        self.df = df

    @staticmethod
    def empty() -> TypeCollection:
        return TypeCollection(
            df=pd.DataFrame(columns=TypeCollectionSchemaColumns).pipe(
                pt.DataFrame[TypeCollectionSchema]
            )
        )

    @staticmethod
    def from_annotations(
        file: pathlib.Path, annotations: Annotations, strict: bool
    ) -> TypeCollection:

        from pandas._libs import missing

        filename = str(file)

        contents = list()
        for fkey, fanno in annotations.functions.items():
            # NOTE: if fanno.returns is None, this is ACCURATE (for Python itself)!,
            # NOTE: However, in strict mode, we take this to be INACCURATE, as our primary objective
            # NOTE: is to denote missing coverage
            contents.append(
                (
                    filename,
                    TypeCollectionCategory.CALLABLE_RETURN,
                    fkey.name,
                    _stringify(fanno.returns) or (missing.NA if strict else "None"),
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
                        TypeCollectionCategory.CALLABLE_PARAMETER,
                        f"{fkey.name}.{param.name.value}",
                        _stringify(param.annotation) or missing.NA,
                    )
                )

        for qname, anno in annotations.attributes.items():
            # NOTE: assignments to variables without an annotation are deemed INVALID
            contents.append(
                (
                    filename,
                    TypeCollectionCategory.VARIABLE,
                    qname,
                    _stringify(anno) or missing.NA,
                )
            )

        for cqname, cdef in annotations.class_definitions.items():
            for stmt in cdef.body.body:
                # NOTE: No need to check for validity of `annassign.annotation`
                # NOTE: as cst.AnnAssign exists precisely so that the Annotation exists
                annassign: cst.AnnAssign = stmt.body[0]
                contents.append(
                    (
                        filename,
                        TypeCollectionCategory.CLASS_ATTR,
                        f"{cqname}.{_stringify(annassign.target)}",
                        _stringify(annassign.annotation),
                    )
                )

        df = pd.DataFrame(contents, columns=TypeCollectionSchemaColumns)
        return TypeCollection(df.pipe(pt.DataFrame[TypeCollectionSchema]))

    @staticmethod
    def load(path: str | pathlib.Path) -> TypeCollection:
        return TypeCollection(
            df=pd.read_csv(
                path,
                sep="\t",
                converters={"category": lambda c: TypeCollectionCategory[c]},
            ).pipe(pt.DataFrame[TypeCollectionSchema])
        )

    def write(self, path: str | pathlib.Path) -> None:
        self.df.to_csv(
            path,
            sep="\t",
            index=False,
            header=TypeCollectionSchemaColumns,
        )

    @pa.check_types
    def update(self, other: pt.DataFrame[TypeCollectionSchema]) -> None:
        self.df = (
            pd.concat([self.df, other], ignore_index=True)
            .drop_duplicates(keep="last")
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )

    def merge_into(self, other: TypeCollection) -> None:
        self.update(other.df)


# Currently schema-less as merging results into unpredictable column naming
class MergedAnnotations:
    @pa.check_types
    def __init__(self, df: pt.DataFrame[MergedAnnotationSchema]) -> None:
        self.df = df

    @staticmethod
    def from_collections(
        collections: list[tuple[pathlib.Path, TypeCollection]]
    ) -> MergedAnnotations:
        renamed_dfs = map(
            lambda pathdf: pathdf[1].df.rename({"anno": f"{pathdf[0].name}_anno"}),
            collections,
        )
        df: pt.DataFrame[MergedAnnotationSchema] = functools.reduce(
            lambda acc, curr: pd.merge(
                left=acc, right=curr, how="outer", on=MergedAnnotationSchemaColumns
            ),
            renamed_dfs,
        ).pipe(pt.DataFrame[MergedAnnotationSchema])

        return MergedAnnotations(df=df)

    @staticmethod
    def load(path: str | pathlib.Path) -> MergedAnnotations:
        return MergedAnnotations(
            df=pd.read_csv(
                path,
                sep="\t",
                converters={"category": lambda c: TypeCollectionCategory[c]},
            ).pipe(pt.DataFrame[MergedAnnotationSchema])
        )

    def write(self, path: str | pathlib.Path) -> None:
        self.df.to_csv(
            path,
            sep="\t",
            index=False,
        )
