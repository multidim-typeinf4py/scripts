import dataclasses
import typing

import libcst
from libcst import matchers as m

from libsa4py.cst_transformers import ParametricTypeDepthReducer

import polars as pl
from polars.type_aliases import IntoExpr
from polars.datatypes.classes import Utf8

from scripts.common.schemas import InferredSchema, TypeCollectionCategory
import libcst


@dataclasses.dataclass
class MappingConfig:
    dequalification: bool = True
    normalise_unions: bool = True
    normalise_typing_imports: bool = True

    reduce_to_parametric: bool = True
    unquote_annotations: bool = True

    uncovered_variables: list = dataclasses.field(
        default_factory=lambda: [
            "Any",
            "None",
            "object",
            "type",
            "Type[Any]",
            "Type[cls]",
            "Type[type]",
            "Type",
            "TypeVar",
            "Optional[Any]",
        ]
    )

    def into_exprs(self) -> list[IntoExpr]:
        queries = list[IntoExpr]()

        if self.unquote_annotations:
            # Remove leading quote signs
            queries.append(_remove_surrounding_quotes())

        if self.dequalification:
            queries.append(_dequalify())

        if self.uncovered_variables:
            queries.append(_mark_useless_variable_annotations(self.uncovered_variables))

        if self.normalise_typing_imports:
            queries.extend(
                _normalise_typing_imports(
                    # aliases={
                    # "(?<=.*)any(?<=.*)|(?<=.*)unknown(?<=.*)": "Any",
                    # "^{}$|^Dict$|^Dict\[\]$|(?<=.*)Dict\[Any, *?Any\](?=.*)|^Dict\[unknown, *Any\]$": "dict",
                    # "^Set$|(?<=.*)Set\[\](?<=.*)|^Set\[Any\]$": "set",
                    # "^Tuple$|(?<=.*)Tuple\[\](?<=.*)|^Tuple\[Any\]$|(?<=.*)Tuple\[Any, *?\.\.\.\](?=.*)|^Tuple\[unknown, *?unknown\]$|^Tuple\[unknown, *?Any\]$|(?<=.*)tuple\[\](?<=.*)": "tuple",
                    # "^Tuple\[(.+), *?\.\.\.\]$": r"Tuple[\1]",
                    # "\\bText\\b": "str",
                    # "^\[\]$|(?<=.*)List\[\](?<=.*)|^List\[Any\]$|^List$": "list",
                    # "^\[{}\]$": "List[dict]",
                    # "(?<=.*)Literal\['.*?'\](?=.*)": "Literal",
                    # "(?<=.*)Literal\[\d+\](?=.*)": "Literal",  # Maybe int?!
                    # "^Callable\[\.\.\., *?Any\]$|^Callable\[\[Any\], *?Any\]$|^Callable[[Named(x, Any)], Any]$": "Callable",
                    # "^Iterator[Any]$": "Iterator",
                    # "^OrderedDict[Any, *?Any]$": "OrderedDict",
                    # "^Counter[Any]$": "Counter",
                    # "(?<=.*)Match[Any](?<=.*)": "Match",
                    # }
                    aliases={
                        "^Dict$": "dict",
                        "Dict\[": "dict[",
                        "^Set$": "set",
                        "Set\[": "set[",
                        "^List$": "list",
                        "List\[": "list[",
                        "^Tuple$": "tuple",
                        "Tuple\[": "tuple[",
                    }
                )
            )

        if self.normalise_unions:
            queries.append(_normalise_unions())

        if self.reduce_to_parametric:
            queries.append(_reduce_parametric())

        return queries


def _remove_surrounding_quotes() -> IntoExpr:
    return pl.col(InferredSchema.anno).str.strip(characters=r"'\"")


def _normalise_typing_imports(aliases: typing.Mapping[str, str]) -> list[IntoExpr]:
    return [
        pl.col(InferredSchema.anno).str.replace_all(pattern=k, value=v) for k, v in aliases.items()
    ]


def _mark_useless_variable_annotations(useless: list[str]) -> IntoExpr:
    return pl.when(
        (pl.col(InferredSchema.category) == str(TypeCollectionCategory.VARIABLE))
        & (pl.col(InferredSchema.anno).is_in(useless))
    ).then(None)


def _dequalify() -> IntoExpr:
    class TypeDequalifier(libcst.CSTTransformer):
        def leave_Attribute(
            self, _: libcst.Attribute, updated_node: libcst.Attribute
        ) -> libcst.Name:
            return updated_node.attr

    return pl.col(InferredSchema.anno).apply(
        lambda a: libcst.parse_module(a).visit(TypeDequalifier()).code,
        return_dtype=Utf8,
    )


def _normalise_unions() -> IntoExpr:
    class UnionNormaliser(libcst.CSTTransformer):
        def leave_BinaryOperation(
            self, original_node: libcst.BinaryOperation, updated_node: libcst.BinaryOperation
        ) -> libcst.BaseExpression:
            if not m.matches(updated_node.operator, m.BitOr()):
                return updated_node

            if m.matches(updated_node.left, m.Name()) and m.matches(updated_node.right, m.Name()):
                return libcst.Subscript(
                    value=libcst.Name("Union"),
                    slice=list(
                        map(libcst.SubscriptElement, [updated_node.left, updated_node.right])
                    ),
                )

            if m.matches(updated_node.left, m.Subscript(value=libcst.Name("Union"))):
                return libcst.Subscript(
                    value=libcst.Name("Union"),
                    slice=list(
                        map(libcst.SubscriptElement, [*updated_node.left.slice, updated_node.right])
                    ),
                )

            raise Exception(f"Unhandled binop: {libcst.Module([original_node]).code}")

        def leave_SubscriptElement(
            self, original_node: libcst.SubscriptElement, updated_node: libcst.SubscriptElement
        ) -> libcst.FlattenSentinel[libcst.SubscriptElement] | libcst.SubscriptElement:
            # print(original_node)

            if m.matches(updated_node.slice, m.Index(value=m.Subscript(value=m.Name("Union")))):
                return libcst.FlattenSentinel(updated_node.slice.value.slice)

            return original_node

    return pl.col(InferredSchema.anno).apply(
        lambda a: libcst.parse_module(a).visit(UnionNormaliser()).code,
        return_dtype=Utf8,
    )


def _reduce_parametric() -> IntoExpr:
    return pl.col(InferredSchema.anno).apply(
        lambda a: libcst.parse_module(a).visit(ParametricTypeDepthReducer(max_annot_depth=2)).code,
        return_dtype=Utf8,
    )
