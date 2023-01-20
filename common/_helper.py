
import libcst as cst
import pandas as pd
import typing

from common.schemas import ContextSymbolSchema


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




def generate_qname_ssas(var_names: pd.Series) -> pd.Series:
    qname_ssa_suffix = var_names.groupby(var_names).cumcount()
    return var_names + "$" + (qname_ssa_suffix + 1).astype(str)