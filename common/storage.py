from __future__ import annotations

import functools
import pathlib
import typing

from libcst.codemod.visitors._apply_type_annotations import Annotations
import libcst as cst

from pandas._libs import missing
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
    """
    Maintain DataFrames used for tracking symbols across multiple files,
    and allow for querying of symbols in a manner that makes comparing occurrences of
    a singular symbol across the "same" file in different places simple
    """

    @pa.check_types
    def __init__(self, df: pt.DataFrame[MergedAnnotationSchema]) -> None:
        self.df = df

    @staticmethod
    def from_collections(
        collections: typing.Sequence[tuple[pathlib.Path, TypeCollection]],
    ) -> MergedAnnotations:
        dfs = map(
            lambda pdf: pdf[1].df.rename(columns={"anno": f"{pdf[0].name}_anno"}),
            collections,
        )

        df: pd.DataFrame = functools.reduce(
            lambda acc, curr: pd.merge(
                left=acc, right=curr, how="outer", on=["file", "category", "qname"]
            ),
            dfs,
        ).fillna(missing.NA)

        return MergedAnnotations(df=df.pipe(pt.DataFrame[MergedAnnotationSchema]))

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

    def differing(
        self,
        *,
        roots: list[pathlib.Path] | None = None,
        files: list[pathlib.Path] | None = None,
    ) -> pt.DataFrame[MergedAnnotationSchema]:

        # Query relevant files
        match files:
            case [pathlib.Path(), *_]:
                sfiles = list(map(str, files))
                relevant_file_df = self.df[self.df["file"].isin(sfiles)]
            case None:
                relevant_file_df = self.df
            case _:
                raise AssertionError(f"Unhandled case for `files`: {list(map(type, files))}")

        # Query relevant projects
        match roots:
            case [pathlib.Path(), *_]:
                relevant_root_df = relevant_file_df.filter(
                    items=[f"{root.name}_anno" for root in roots]
                )
            case None:
                relevant_root_df = relevant_file_df.filter(regex=r"_anno$")
            case _:
                raise AssertionError(f"Unhandled case for `roots`: {list(map(type, roots))}")

        # Compute differences between projects
        relevant_anno_df = relevant_root_df[
            relevant_root_df.apply(lambda row: pd.Series.nunique(row, dropna=False), axis=1) != 1
        ]

        anno_diff = pd.merge(
            left=self.df[MergedAnnotationSchemaColumns],
            right=relevant_anno_df,
            left_index=True,
            right_index=True,
        )
        return anno_diff.pipe(pt.DataFrame[MergedAnnotationSchema])

    def hints_for_repo(
        self, *, repo: pathlib.Path, files: list[pathlib.Path] | None = None
    ) -> pt.DataFrame[TypeCollectionSchema]:
        df = self.df[MergedAnnotationSchemaColumns + [f"{repo.name}_anno"]].rename(
            columns={f"{repo.name}_anno": "anno"}
        )

        sfiles = list(map(str, files or []))
        file_df = df[df["file"].isin(sfiles)]

        return file_df.pipe(pt.DataFrame[TypeCollectionSchema])
