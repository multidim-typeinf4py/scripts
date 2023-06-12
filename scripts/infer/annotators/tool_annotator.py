from __future__ import annotations


import abc
import dataclasses
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

from ..normalisers import bracket as b, typing_aliases as t, union as u

T = typing.TypeVar("T")
U = typing.TypeVar("U")


@dataclasses.dataclass
class Normalisation:
    bad_list_generics: bool = False
    bad_tuple_generics: bool = False
    bad_dict_generics: bool = False

    lowercase_aliases: bool = False

    unnest_union_t: bool = False
    union_or_to_union_t: bool = False

    typing_text_to_str: bool = False

    outer_optional_to_t: bool = False
    outer_final_to_t: bool = False


class ParallelTopNAnnotator(codemod.ContextAwareTransformer, abc.ABC, typing.Generic[T, U]):
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

    def transform_module(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        path = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        predictions = self.extract_predictions_for_file(self.path2batches, path, self.topn)
        annotator = self.annotator(predictions)

        annotated = annotator.transform_module(tree)

        normalisation = self.normalisation()
        if normalisation.bad_list_generics:
            annotated = b.SquareBracketsToList(context=self.context).transform_module(annotated)

        if normalisation.bad_tuple_generics:
            annotated = b.RoundBracketsToTuple(context=self.context).transform_module(annotated)

        if normalisation.bad_dict_generics:
            annotated = b.CurlyBracesToDict(context=self.context).transform_module(annotated)

        if normalisation.lowercase_aliases:
            annotated = t.LowercaseTypingAliases(context=self.context).transform_module(annotated)

        if normalisation.typing_text_to_str:
            annotated = t.TextToStr(context=self.context).transform_module(annotated)

        if normalisation.outer_optional_to_t:
            annotated = t.RemoveOuterOptional(context=self.context).transform_module(annotated)

        if normalisation.outer_final_to_t:
            annotated = t.RemoveOuterFinal(context=self.context).transform_module(annotated)

        if normalisation.unnest_union_t:
            annotated = u.Flatten(context=self.context).transform_module(annotated)

        if normalisation.union_or_to_union_t:
            annotated = u.Pep604(context=self.context).transform_module(annotated)

        return annotated

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
