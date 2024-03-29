from __future__ import annotations

import itertools
import pathlib

from src.common.annotations import (
    MultiVarAnnotations,
)
from libcst.codemod.visitors._apply_type_annotations import (
    Annotations,
    FunctionKey,
    FunctionAnnotation,
)

import libcst as cst

import pandas as pd
import pandera as pa
import pandera.typing as pt


from .ast_helper import _stringify, generate_qname_ssas_for_project
from .schemas import (
    TypeCollectionCategory,
    TypeCollectionSchema,
)


class TypeCollection:
    @pa.check_types
    def __init__(self, df: pt.DataFrame[TypeCollectionSchema]) -> None:
        self.df = df

    @staticmethod
    def empty() -> TypeCollection:
        return TypeCollection(df=TypeCollectionSchema.example(size=0))

    @staticmethod
    def from_annotations(
        file: pathlib.Path, annos: MultiVarAnnotations, strict: bool
    ) -> TypeCollection:
        from pandas._libs import missing

        filename = str(file)

        contents = list()
        for fkey, fanno in annos.functions.items():
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

        for qname, annos in annos.attributes.items():
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

        cs = [
            TypeCollectionSchema.file,
            TypeCollectionSchema.category,
            TypeCollectionSchema.qname,
            TypeCollectionSchema.anno,
        ]
        df = pd.DataFrame(contents, columns=cs)

        return TypeCollection(
            df.pipe(generate_qname_ssas_for_project).pipe(
                pt.DataFrame[TypeCollectionSchema]
            )
        )

    @staticmethod
    def to_libcst_annotations(
        collection: TypeCollection | pt.DataFrame[TypeCollectionSchema],
        baseline: pt.DataFrame[TypeCollectionSchema],
    ) -> Annotations:
        """Create a LibCST Annotations object from the provided DataFrame.
        NOTE: The keys of this Annotations object are QNAME_SSAs, not QNAMEs!
        """

        df = collection.df if isinstance(collection, TypeCollection) else collection

        # Add missing symbols and order by baseline
        ordered = pd.merge(
            left=df,
            right=baseline.drop(columns=["anno"]),
            how="right",
            on=[
                TypeCollectionSchema.file,
                TypeCollectionSchema.category,
                TypeCollectionSchema.qname_ssa,
            ],
            sort=False,
        )

        # Make sure symbols that are not in the baseline, but could be inferred
        df = pd.concat([ordered, df], ignore_index=True).drop_duplicates(
            subset=[
                TypeCollectionSchema.file,
                TypeCollectionSchema.category,
                TypeCollectionSchema.qname_ssa,
            ],
            keep="first",
        )

        dups = df.duplicated(
            subset=[
                TypeCollectionSchema.file,
                TypeCollectionSchema.category,
                TypeCollectionSchema.qname_ssa,
            ],
            keep=False,
        )
        if dups.any():
            raise RuntimeError(
                "Cannot annotate source code when conflicts have not been resolved (this includes remnants of top-n "
                "predictions!)"
            )

        def functions() -> dict[FunctionKey, FunctionAnnotation]:
            fs: dict[FunctionKey, FunctionAnnotation] = {}

            fs_df = df[
                df[TypeCollectionSchema.category]
                == TypeCollectionCategory.CALLABLE_RETURN
            ]
            param_df = df[
                df[TypeCollectionSchema.category]
                == TypeCollectionCategory.CALLABLE_PARAMETER
            ]

            sep_df = param_df[TypeCollectionSchema.qname_ssa].str.rsplit(
                pat=".", n=1, expand=True
            )

            if sep_df.empty:
                sep_df = pd.DataFrame(columns=["fname", "argname"])
            else:
                sep_df = sep_df.set_axis(["fname", "argname"], axis=1)
            param_df = pd.merge(param_df, sep_df, left_index=True, right_index=True)

            # Use function name to find parameters; parameters cannot exist without functions
            # but functions without parameters cannot exist
            for fname in fs_df[TypeCollectionSchema.qname_ssa]:
                select_params = param_df[param_df["fname"] == fname]
                rettype = fs_df[fs_df[TypeCollectionSchema.qname_ssa] == fname]
                if len(rettype):
                    rettype_anno = rettype[TypeCollectionSchema.anno].iloc[0]
                else:
                    rettype_anno = None
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

                fa_params = cst.Parameters(params)
                fa_returns = (
                    cst.Annotation(cst.parse_expression(rettype_anno))
                    if pd.notna(rettype_anno)
                    else None
                )
                anno = FunctionAnnotation(
                    parameters=fa_params,
                    returns=fa_returns,
                )
                fs[key] = anno

            return fs

        def variables() -> dict[str, cst.Annotation]:
            vs: dict[str, cst.Annotation] = {}
            var_df = df[
                df[TypeCollectionSchema.category] == TypeCollectionCategory.VARIABLE
            ]

            for qname, anno in var_df[
                [TypeCollectionSchema.qname_ssa, TypeCollectionSchema.anno]
            ].itertuples(index=False):
                if pd.notna(anno):
                    vs[qname] = cst.Annotation(cst.parse_expression(anno))

            return vs

        annos = Annotations(
            functions=functions(),
            attributes=variables(),
            class_definitions=dict(),
            typevars=dict(),
            names=set(),
        )
        return annos

    @staticmethod
    def load(path: str | pathlib.Path) -> TypeCollection:
        return TypeCollection(
            df=pd.read_csv(
                path,
                converters={"category": lambda c: TypeCollectionCategory[c]},
            ).pipe(pt.DataFrame[TypeCollectionSchema])
        )

    def write(self, path: str | pathlib.Path) -> None:
        self.df.to_csv(
            path,
            index=False,
        )

    @pa.check_types
    def update(self, other: pt.DataFrame[TypeCollectionSchema]) -> None:
        self.df = pd.concat([self.df, other], ignore_index=True).pipe(
            pt.DataFrame[TypeCollectionSchema]
        )

    def merge_into(self, other: TypeCollection) -> None:
        self.update(other.df)
