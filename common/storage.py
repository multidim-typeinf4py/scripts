from __future__ import annotations
from collections import defaultdict

import functools
import itertools
import pathlib
import typing

from common.annotations import (
    MultiVarAnnotations,
)
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)
import libcst as cst

from pandas._libs import missing
import pandas as pd
import pandera as pa
import pandera.typing as pt


from ._helper import _stringify, generate_qname_ssas_for_project
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
        file: pathlib.Path, annotations: MultiVarAnnotations, strict: bool
    ) -> TypeCollection:
        from pandas._libs import missing

        filename = str(file)

        contents = list()
        for fkey, fanno in annotations.functions.items():
            # NOTE: if fanno.returns is None, this is ACCURATE (for Python itself)!,
            # NOTE: However, in strict mode, we take this to be INACCURATE, as our primary objective
            # NOTE: is to denote missing coverage
            qname = fkey.name
            contents.append(
                (
                    filename,
                    TypeCollectionCategory.CALLABLE_RETURN,
                    qname,
                    _stringify(fanno.returns) or (missing.NA if strict else "None"),
                )
            )

            # NOTE: if param.annotation is None, this is NOT accurate!, as:
            # NOTE: unlike functions, no assumption of None is given
            # NOTE: therefore, we must differentiate between "None" and None, and mark
            # NOTE: the latter as INVALID!
            for param in itertools.chain(
                fanno.parameters.posonly_params,
                fanno.parameters.params,
                fanno.parameters.kwonly_params,
            ):
                qname = f"{fkey.name}.{param.name.value}"
                contents.append(
                    (
                        filename,
                        TypeCollectionCategory.CALLABLE_PARAMETER,
                        qname,
                        _stringify(param.annotation) or missing.NA,
                    )
                )

        for qname, annos in annotations.attributes.items():
            # NOTE: assignments to variables without an annotation are deemed INVALID
            if annos:
                for anno in annos:
                    contents.append(
                        (
                            filename,
                            TypeCollectionCategory.VARIABLE,
                            qname,
                            _stringify(anno) or missing.NA,
                        )
                    )

            else:
                contents.append(
                    (
                        filename,
                        TypeCollectionCategory.VARIABLE,
                        qname,
                        missing.NA,
                    )
                )

        for cqname, cdef in annotations.class_definitions.items():
            for stmt in cdef.body.body:
                # NOTE: No need to check for validity of `annassign.annotation`
                # NOTE: as cst.AnnAssign exists precisely so that the Annotation exists
                if isinstance(annassign := stmt.body[0], cst.AnnAssign):
                    qname = f"{cqname}.{_stringify(annassign.target)}"
                    assert (anno := _stringify(annassign.annotation)) is not None

                    contents.append(
                        (
                            filename,
                            TypeCollectionCategory.CLASS_ATTR,
                            qname,
                            anno,
                        )
                    )

        cs = [c for c in TypeCollectionSchemaColumns if c != TypeCollectionSchema.qname_ssa]
        df = pd.DataFrame(contents, columns=cs)

        wqname_ssas: pd.DataFrame = df.pipe(generate_qname_ssas_for_project)
        return TypeCollection(wqname_ssas.pipe(pt.DataFrame[TypeCollectionSchema]))

    @staticmethod
    def to_libcst_annotations(
        collection: TypeCollection | pt.DataFrame[TypeCollectionSchema],
    ) -> Annotations:
        """Create a LibCST Annotations object from the provided DataFrame.
        NOTE: The keys of this Annotations object are QNAME_SSAs, not QNAMEs!"""
        df = collection.df if isinstance(collection, TypeCollection) else collection
        dups = df.duplicated(
            subset=[
                TypeCollectionSchema.file,
                TypeCollectionSchema.category,
                TypeCollectionSchema.qname_ssa,
                TypeCollectionSchema.anno,
            ],
            keep=False,
        )
        if dups.any():
            raise RuntimeError(
                "Cannot annotate source code when conflicts have not been resolved (this includes remnants of top-n predictions!)"
            )

        def functions() -> dict[FunctionKey, FunctionAnnotation]:
            fs: dict[FunctionKey, FunctionAnnotation] = {}

            fs_df = df[df[TypeCollectionSchema.category] == TypeCollectionCategory.CALLABLE_RETURN]
            param_df = df[
                df[TypeCollectionSchema.category] == TypeCollectionCategory.CALLABLE_PARAMETER
            ]

            sep_df = param_df[TypeCollectionSchema.qname_ssa].str.rsplit(pat=".", n=1, expand=True)
            if sep_df.empty:
                return fs
            sep_df = sep_df.set_axis(["fname", "argname"], axis=1)
            param_df = pd.merge(param_df, sep_df, left_index=True, right_index=True)

            for fname, rettype in fs_df[
                [TypeCollectionSchema.qname_ssa, TypeCollectionSchema.anno]
            ].itertuples(index=False):
                select_params = param_df[param_df["fname"] == fname]
                params = [
                    cst.Param(
                        name=cst.Name(value),
                        annotation=cst.Annotation(cst.parse_expression(anno))
                        if pd.notna(anno)
                        else None,
                    )
                    for value, anno in select_params[
                        ["argname", TypeCollectionSchema.anno]
                    ].itertuples(index=False)
                ]

                key = FunctionKey.make(name=fname, params=cst.Parameters(params))
                anno = FunctionAnnotation(
                    parameters=cst.Parameters(params),
                    returns=cst.Annotation(cst.parse_expression(rettype))
                    if pd.notna(rettype)
                    else None,
                )
                fs[key] = anno

            return fs

        def variables() -> dict[str, list[cst.Annotation]]:
            vs: dict[str, list[cst.Annotation]] = {}
            var_df = df[df[TypeCollectionSchema.category] == TypeCollectionCategory.VARIABLE]

            for qname, anno in var_df[
                [TypeCollectionSchema.qname_ssa, TypeCollectionSchema.anno]
            ].itertuples(index=False):
                if pd.notna(anno):
                    vs[qname] = cst.Annotation(cst.parse_expression(anno))

            return vs

        def attributes() -> dict[str, cst.ClassDef]:
            attrs: dict[str, cst.ClassDef] = {}
            attr_df = df[df[TypeCollectionSchema.category] == TypeCollectionCategory.CLASS_ATTR]

            sep_df = attr_df[TypeCollectionSchema.qname_ssa].str.rsplit(pat=".", n=1, expand=True)
            if sep_df.empty:
                return attrs
            sep_df = sep_df.set_axis(["cqname", "attrname"], axis=1)
            attr_df = pd.merge(attr_df, sep_df, left_index=True, right_index=True)

            for cqname, group in attr_df.groupby(by="cqname"):
                *_, cname = cqname.split(".")
                hints = [
                    cst.AnnAssign(
                        target=cst.Name(aname),
                        annotation=cst.Annotation(cst.parse_expression(hint)),
                    )
                    for aname, hint in group[["attrname", TypeCollectionSchema.anno]].itertuples(
                        index=False
                    )
                    if pd.notna(hint)
                ]
                attrs[cqname] = cst.ClassDef(
                    name=cst.Name(cname),
                    body=cst.IndentedBlock(body=[cst.SimpleStatementSuite(body=hints)]),
                )

            return attrs

        return Annotations(
            functions=functions(),
            attributes=variables(),
            class_definitions=attributes(),
            typevars=dict(),
            names=set(),
        )

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
        self.df = pd.concat([self.df, other], ignore_index=True).pipe(
            pt.DataFrame[TypeCollectionSchema]
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
                left=acc,
                right=curr,
                how="outer",
                on=[
                    TypeCollectionSchema.file,
                    TypeCollectionSchema.category,
                    TypeCollectionSchema.qname_ssa,
                ],
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
