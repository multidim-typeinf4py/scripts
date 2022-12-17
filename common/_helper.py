import libcst as cst

import typing


@typing.no_type_check
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

    