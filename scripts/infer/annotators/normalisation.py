import dataclasses
import functools

import libcst
from libcst import codemod

from scripts.infer.normalisers import (
    bad_generics as b,
    typing_aliases as t,
    union as u,
    unstringify as us,
    parametric_depth as p,
    remove_anys as r
)


@dataclasses.dataclass
class Normalisation:
    # typing.Type["AbstractExtractors"] -> typing.Type[AbstractExecutors]
    unquote: bool = True

    # typing.Text -> str
    typing_text_to_str: bool = True

    # List[List[Tuple[int]]] -> List[List[Any]]
    limit_parametric_depth: bool = True

    # [] -> List, (str, int) -> Tuple[str, int], {} -> dict, # (builtins?).{False, True} -> bool
    bad_generics: bool = False

    # {list, tuple, dict} -> typing.{List, Tuple, Dict}
    # uppercase_aliases: bool = False

    # Optional[T] -> Union[T,None],
    # int | str -> Union[int, str]
    # Union[Union[int]] -> Union[int]
    # + sorting
    normalise_union_ts: bool = False

    # If all type arguments are Any, drop them all. e.g., rewrite List[Any] to List
    remove_if_all_any: bool = False

    # (typing?).{List, Tuple, Dict} -> {list, tuple, dict}
    lowercase_aliases: bool = False


    # Optional[T] -> T
    # outer_optional_to_t: bool = False

    # Final[T] -> T
    # outer_final_to_t: bool = False

    def transformers(
        self, context: codemod.CodemodContext
    ) -> list[codemod.Codemod]:
        # assert self.lowercase_aliases + self.uppercase_aliases <= 1

        ts = list[codemod.Codemod]()
        if self.unquote:
            ts.append(us.Unquote(context=context))

        if self.typing_text_to_str:
            ts.append(t.TextToStr(context=context))

        if self.limit_parametric_depth:
            ts.append(p.ParametricTypeDepthReducer(context=context))

        if self.bad_generics:
            ts.append(b.BadGenericsNormaliser(context=context))

        #if self.outer_optional_to_t:
        #    ts.append(t.RemoveOuterOptional(context=context))

        #if self.outer_final_to_t:
        #    ts.append(t.RemoveOuterFinal(context=context))

        if self.normalise_union_ts:
            ts.append(u.UnionNormaliser(context=context))

        if self.remove_if_all_any:
            ts.append(r.RemoveAnys(context=context))

        if self.lowercase_aliases:
            ts.append(t.LowercaseTypingAliases(context=context))

        return ts


class Normaliser(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, strategy: Normalisation) -> None:
        super().__init__(context=context)
        self.strategy = strategy

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        transformers = self.strategy.transformers(context=self.context)
        return functools.reduce(
            lambda mod, trans: trans.transform_module(mod),
            transformers,
            tree
        )