from __future__ import annotations

import abc
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

from .normalisation import Normalisation, Normaliser

T = typing.TypeVar("T")
U = typing.TypeVar("U")


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

        normalised = Normaliser(
            context=self.context, strategy=self.normalisation()
        ).transform_module(annotated)
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
            #with utils.scratchpad(project) as sc:
            anno = codemod.parallel_exec_transform_with_prettyprint(
                transform=cls(
                    context=codemod.CodemodContext(),
                    paths2topn=predictions,
                    topn=n - 1,
                    **kwargs,
                ),
                jobs=utils.worker_count(),
                repo_root=str(project),
                files=[str(project / s) for s in subset],
            )
            anno_res = utils.format_parallel_exec_result(
                f"Annotated with {tool.method()} @ topn={n}", result=anno
            )
            tool.logger.info(anno_res)

            collection = build_type_collection(
                root=project, allow_stubs=False, subset=subset
            ).df.assign(topn=n)
            collections.append(collection)
        return (
            pd.concat(collections, ignore_index=True)
            .assign(method=tool.method())
            .pipe(pt.DataFrame[InferredSchema])
        )
