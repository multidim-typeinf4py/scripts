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


def _reduce_parametric() -> IntoExpr:
    return pl.col(InferredSchema.anno).apply(
        lambda a: libcst.parse_module(a).visit(ParametricTypeDepthReducer(max_annot_depth=2)).code,
        return_dtype=Utf8,
    )
