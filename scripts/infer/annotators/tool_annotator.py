from __future__ import annotations

import abc
import dataclasses
import functools
import pathlib
import typing

import libcst
import pandas as pd
from libcst import codemod
from pandera import typing as pt

from scripts import utils
from scripts.common.schemas import InferredSchema
from scripts.infer.inference import Inference
from scripts.symbols.collector import build_type_collection
from scripts.infer.normalisers import bracket as b, typing_aliases as t, union as u, literal_to_base as l

T = typing.TypeVar("T")
U = typing.TypeVar("U")


@dataclasses.dataclass
class Normalisation:
    # [] -> List
    bad_list_generics: bool = False

    # (str, int) -> Tuple[str, int]
    bad_tuple_generics: bool = False

    # {} -> dict
    bad_dict_generics: bool = False

    # (builtins?).{False, True} -> bool
    bad_literals: bool = False

    # (typing?).{List, Tuple, Dict} -> {list, tuple, dict}
    lowercase_aliases: bool = False

    # Union[Union[int]] -> Union[int]
    unnest_union_t: bool = False

    # int | str -> Union[int, str]
    union_or_to_union_t: bool = False

    # typing.Text -> str
    typing_text_to_str: bool = False

    # Optional[T] -> T
    outer_optional_to_t: bool = False

    # Final[T] -> T
    outer_final_to_t: bool = False

    def transformers(self, context: codemod.CodemodContext) -> list[codemod.ContextAwareTransformer]:
        ts = list[codemod.ContextAwareTransformer]()
        if self.bad_list_generics:
            ts.append(b.SquareBracketsToList(context=context))

        if self.bad_tuple_generics:
            ts.append(b.RoundBracketsToTuple(context=context))

        if self.bad_dict_generics:
            ts.append(b.CurlyBracesToDict(context=context))

        if self.bad_literals:
            ts.append(l.LiteralToBaseClass())

        if self.lowercase_aliases:
            ts.append(t.LowercaseTypingAliases(context=context))

        if self.typing_text_to_str:
            ts.append(t.TextToStr(context=context))

        if self.outer_optional_to_t:
            ts.append(t.RemoveOuterOptional(context=context))

        if self.outer_final_to_t:
            ts.append(t.RemoveOuterFinal(context=context))

        if self.unnest_union_t:
            ts.append(u.Flatten(context=context))

        if self.union_or_to_union_t:
            ts.append(u.Pep604(context=context))

        return ts


class ParallelTopNAnnotator(codemod.Codemod, abc.ABC, typing.Generic[T, U]):
    def __init__(
        self,
        context: codemod.CodemodContext,
        paths2topn: T,
        topn: int,
    ) -> None:
        super().__init__(context)
        self.path2batches = paths2topn
        self.topn = topn

    @abc.abstractmethod
    def extract_predictions_for_file(self, path2topn: T, path: pathlib.Path, topn: int) -> U:
        ...

    @abc.abstractmethod
    def annotator(self, annotations: U) -> codemod.Codemod:
        ...

    @abc.abstractmethod
    def normalisation(self) -> Normalisation:
        ...

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        path = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        predictions = self.extract_predictions_for_file(self.path2batches, path, self.topn)
        annotator = self.annotator(predictions)

        annotated = annotator.transform_module(tree)

        normalisation = self.normalisation()

        return functools.reduce(
            lambda mod, transformer: transformer.transform_module(mod),
            normalisation.transformers(self.context),
            annotated,
        )

    @classmethod
    def collect_topn(
        cls,
        project: pathlib.Path,
        subset: set[pathlib.Path],
        predictions: T,
        topn: int,
        tool: Inference,
        **kwargs,
    ) -> pt.DataFrame[InferredSchema]:
        collections = [InferredSchema.example(size=0)]
        for n in range(1, topn + 1):
            with utils.scratchpad(project) as sc:
                anno = codemod.parallel_exec_transform_with_prettyprint(
                    transform=cls(
                        context=codemod.CodemodContext(),
                        paths2topn=predictions,
                        topn=n - 1,
                        **kwargs,
                    ),
                    jobs=utils.worker_count(),
                    repo_root=str(sc),
                    files=[sc / s for s in subset],
                )
                anno_res = utils.format_parallel_exec_result(
                    f"Annotated with {tool.method()} @ topn={n}", result=anno
                )
                tool.logger.info(anno_res)

                collection = build_type_collection(
                    root=sc, allow_stubs=False, subset=subset
                ).df.assign(topn=n)
                collections.append(collection)
        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=tool.method())
            .pipe(pt.DataFrame[InferredSchema])
        )
