import libcst as cst

import typing


@typing.no_type_check
def _stringify(node: cst.CSTNode | None) -> str | None:
    if isinstance(node, cst.Annotation):
        if isinstance(node.annotation, cst.Name):
            return node.annotation.value
        elif isinstance(node.annotation, cst.Module):
            return node.annotation.code

    elif isinstance(node, cst.Name):
        return node.value

    elif node is None:
        return None

    raise AssertionError(f"Unhandled node: {node}")
