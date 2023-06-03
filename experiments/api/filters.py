import dataclasses

import polars as pl
from polars.type_aliases import IntoExpr

from scripts.common.schemas import InferredSchema, TypeCollectionCategory


@dataclasses.dataclass
class FilteringConfig:
    drop_trivial_functions: list = dataclasses.field(
        default_factory=lambda: [
            f"__{f}__"
            for f in (
                "str",
                "unicode",
                "repr",
                "len",
                "doc",
                "sizeof",
            )
        ]
    )

    def into_exprs(self) -> list[IntoExpr]:
        queries = []

        if self.drop_trivial_functions:
            queries.append(_filter_trivial_functions(self.drop_trivial_functions))

        return queries


def _filter_trivial_functions(function_names: list[str]) -> IntoExpr:
    functions = pl.col(InferredSchema.category) == str(TypeCollectionCategory.CALLABLE_RETURN)
    trivials = pl.col(InferredSchema.qname).str.split(by=".").list.get(-1).is_in(function_names)
    return ~(functions & trivials)
