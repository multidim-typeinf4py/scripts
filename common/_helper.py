
import libcst as cst
import pandas as pd
import typing

from common.schemas import ContextSymbolSchema, TypeCollectionCategory


def _stringify(node: cst.CSTNode | None) -> str | None:
    if node is None:
        return None

    try:
        return cst.Module([]).code_for_node(node)

    except SyntaxError:
        match node:
            case cst.Annotation():
                return _stringify(node.annotation)
            case _:
                raise AssertionError(f"Unhandled node: {node}")




def generate_var_qname_ssas(var_names: pd.Series) -> pd.Series:
    qname_ssa_suffix = var_names.groupby(var_names).cumcount()
    return var_names + "λ" + (qname_ssa_suffix + 1).astype(str)

def generate_qname_ssas_for_file(df: pd.DataFrame) -> pd.DataFrame:
    # Create qname_ssas
    variables = df[ContextSymbolSchema.category] == TypeCollectionCategory.VARIABLE
    df.loc[~variables, ContextSymbolSchema.qname_ssa] = df.loc[
        ~variables, ContextSymbolSchema.qname
    ]

    df.loc[variables, ContextSymbolSchema.qname_ssa] = generate_var_qname_ssas(
        df.loc[variables, ContextSymbolSchema.qname]
    )
    return df

def generate_qname_ssas_for_project(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(by=ContextSymbolSchema.file, sort=False, group_keys=True).apply(generate_qname_ssas_for_file)