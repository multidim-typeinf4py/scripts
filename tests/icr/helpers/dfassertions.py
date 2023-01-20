import pandas as pd
import pandera.typing as pt

from common.schemas import InferredSchema, TypeCollectionCategory


def _assert(needle: pd.DataFrame, haystack: pt.DataFrame[InferredSchema], check_anno: bool):
    if check_anno:
        haystack = haystack.drop(columns=["file"])
    else:
        haystack = haystack.drop(columns=["file", "anno"])

    common = pd.merge(left=needle, right=haystack, how="left", indicator="indic")
    success = (common["indic"] == "both").all()
    if not success:
        print("WARNING! missing", common[common["indic"] != "both"], sep="\n")

    return success


def _ctor_df(category: TypeCollectionCategory, qname: str, anno: str | None = None) -> pd.DataFrame:
    single = pd.DataFrame(
        {
            "category": [category],
            "qname": [qname],
        }
    )
    if anno is None:
        return single

    return single.assign(anno=anno)


def has_callable(df: pt.DataFrame[InferredSchema], f_qname: str, anno: str | None = None) -> bool:
    single = _ctor_df(category=TypeCollectionCategory.CALLABLE_RETURN, qname=f_qname, anno=anno)
    return _assert(single, df, check_anno=anno is not None)


def has_parameter(
    df: pt.DataFrame[InferredSchema], f_qname: str, arg_name: str, anno: str | None = None
) -> bool:
    single = _ctor_df(
        category=TypeCollectionCategory.CALLABLE_PARAMETER, qname=f"{f_qname}.{arg_name}", anno=anno
    )
    return _assert(single, df, check_anno=anno is not None)


def has_variable(df: pt.DataFrame[InferredSchema], var_qname: str, anno: str | None = None) -> bool:
    single = _ctor_df(category=TypeCollectionCategory.VARIABLE, qname=var_qname, anno=anno)
    return _assert(single, df, check_anno=anno is not None)
