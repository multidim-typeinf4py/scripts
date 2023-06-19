import libcst
from libcst import matchers as m
import pandas as pd


def make_parametric(annotation: str | None) -> str | None:
    class ToParametricTransformer(libcst.CSTTransformer):
        def leave_Subscript(
            self, original_node: libcst.Subscript, updated_node: libcst.Subscript
        ) -> libcst.BaseExpression:
            return updated_node.value
    if pd.isna(annotation) or annotation == "":
        return None

    return libcst.parse_module(annotation).visit(ToParametricTransformer()).code


def is_simple_or_complex(annotation: str | None) -> str | None:
    if pd.isna(annotation) or annotation == "":
        return None

    class Dequalifier(libcst.CSTTransformer):
        def __init__(self) -> None:
            super().__init__()

        def leave_Attribute(self, original_node: libcst.Attribute, updated_node: libcst.Attribute) -> libcst.Name:
            return updated_node.attr

    class ComplexityCounter(m.MatcherDecoratableVisitor):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def visit_Name(self, node: libcst.Name) -> None:
            self.counter += 1

        def visit_Attribute(self, node: libcst.Attribute) -> None:
            self.counter += 1

    visitor = ComplexityCounter()
    libcst.parse_expression(annotation).visit(Dequalifier()).visit(visitor)

    return "simple" if visitor.counter <= 1 else "complex"
