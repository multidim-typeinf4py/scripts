import libcst as cst

import typing


@typing.no_type_check
def _stringify(node: cst.CSTNode | None) -> str | None:
    match node:
        case cst.Annotation(cst.Name(name)):
            return name
        case cst.Annotation(cst.Module()):
            m: cst.Module = node.annotation
            return m.code
        case cst.Name(name):
            return name
        case None:
            return None
        case _:
            raise AssertionError(f"Unhandled node: {node}")
