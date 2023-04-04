import libcst as cst
import pandas as pd
import typing

from common.schemas import ContextSymbolSchema, InferredSchema, SymbolSchema, TypeCollectionCategory


def _stringify(node: cst.CSTNode | None) -> str | None:
    if node is None:
        return None

    try:
        return cst.Module([]).code_for_node(node)

    except SyntaxError:
        if isinstance(node, cst.Annotation):
            return _stringify(node.annotation)
        else:
            raise AssertionError(f"Unhandled node: {node}")


def _generate_var_qname_ssas_for_qname(var_names: pd.Series) -> pd.Series:
    qname_ssa_suffix = var_names.groupby(var_names).cumcount()
    return var_names + "λ" + (qname_ssa_suffix + 1).astype(str)


def _generate_var_qname_ssas_for_topn_qname(df: pd.DataFrame) -> pd.Series:
    qname_ssa_suffix = df.groupby(by=[SymbolSchema.qname, InferredSchema.topn]).cumcount()
    return df[SymbolSchema.qname] + "λ" + (qname_ssa_suffix + 1).astype(str)


def generate_qname_ssas_for_file(df: pd.DataFrame) -> pd.DataFrame:
    # Create qname_ssas
    variables = df[SymbolSchema.category] == TypeCollectionCategory.VARIABLE
    df.loc[~variables, SymbolSchema.qname_ssa] = df.loc[~variables, SymbolSchema.qname]

    if InferredSchema.topn in df.columns:
        df.loc[variables, SymbolSchema.qname_ssa] = _generate_var_qname_ssas_for_topn_qname(
            df.loc[variables, [SymbolSchema.qname, InferredSchema.topn]]
        )
    else:
        df.loc[variables, SymbolSchema.qname_ssa] = _generate_var_qname_ssas_for_qname(
            df.loc[variables, SymbolSchema.qname]
        )
    return df


def generate_qname_ssas_for_project(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df[SymbolSchema.qname_ssa] = pd.Series(dtype="str")
        return df
    return df.groupby(by=SymbolSchema.file, sort=False, group_keys=True).apply(
        generate_qname_ssas_for_file
    )
