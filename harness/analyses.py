import dataclasses

from common.schemas import TypeCollectionSchema

import pandas as pd
import pandera.typing as pt


def full_analysis(
    baseline: pt.DataFrame[TypeCollectionSchema], inferred: pt.DataFrame[TypeCollectionSchema]
) -> None:
    ...


@dataclasses.dataclass
class SymbolCommanality:
    common: pt.DataFrame[TypeCollectionSchema]
    only_in_baseline: pt.DataFrame[TypeCollectionSchema]
    only_in_inferred: pt.DataFrame[TypeCollectionSchema]


def symbol_commonality(
    baseline: pt.DataFrame[TypeCollectionSchema], inferred: pt.DataFrame[TypeCollectionSchema]
) -> SymbolCommanality:
    indic = "commonality"

    comparison = pd.merge(
        baseline,
        inferred,
        on=[
            TypeCollectionSchema.file,
            TypeCollectionSchema.category,
            TypeCollectionSchema.qname,
            TypeCollectionSchema.qname_ssa,
            # TypeCollectionSchema.
        ],
        how="outer",
        indicator=indic,
    )

    return SymbolCommanality(
        common=comparison.loc[comparison[indic] == "both"],
        only_in_baseline=comparison.loc[comparison[indic] == "left_only"],
        only_in_inferred=comparison.loc[comparison[indic] == "right_only"],
    )


@dataclasses.dataclass
class AnnotationCommonality:
    missing: pt.DataFrame[TypeCollectionSchema]
    exact_match: pt.DataFrame[TypeCollectionSchema]
    parametric_match: pt.DataFrame[TypeCollectionSchema]
    differing: pt.DataFrame[TypeCollectionSchema]


def annotation_commonality(
    baseline: pt.DataFrame[TypeCollectionSchema], inferred: pt.DataFrame[TypeCollectionSchema]
) -> AnnotationCommonality:
    comparison = pd.merge(
        baseline,
        inferred,
        on=[
            TypeCollectionSchema.file,
            TypeCollectionSchema.category,
            TypeCollectionSchema.qname,
            TypeCollectionSchema.qname_ssa,
        ],
        how="left",
        suffixes=("_gt", "_it"),
    )

    anno_gt = f"{TypeCollectionSchema.anno}_gt"
    anno_it = f"{TypeCollectionSchema.anno}_it"

    missing = comparison.loc[comparison[anno_it].isna()]
    not_missing = comparison.loc[comparison[anno_gt].notna() & comparison[anno_it].notna()]

    exact_match = not_missing[anno_gt] == not_missing[anno_it]

    def parametric(anno_gt: str, anno_it: str) -> bool:
        return anno_gt == anno_it

    # TODO: use parametric conversion from type4py / libsa4py
    parametric_match = exact_match

    differing_match = ~exact_match & ~parametric_match

    return AnnotationCommonality(
        missing=missing,
        exact_match=not_missing.loc[exact_match],
        parametric_match=not_missing.loc[parametric_match],
        differing=not_missing.loc[differing_match],
    )
