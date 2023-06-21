import pathlib
import sys
import typing

import libcst
from type_check import annotater
from dpu_utils.utils import RichPath
from libcst import codemod, ParserSyntaxError

from scripts.infer.annotators import ParallelTopNAnnotator
from .normalisation import Normalisation


class TypilusPrediction(typing.TypedDict):
    annotation_type: str
    location: tuple[int, int]
    name: str
    node_id: int
    original_annotation: str
    predicted_annotation_logprob_dist: list[tuple[str, float]]
    provenance: str


class TypilusProjectApplier(ParallelTopNAnnotator[RichPath, RichPath]):
    def __init__(
        self,
        context: codemod.CodemodContext,
        paths2topn: RichPath,
        topn: int,
        typing_rules: pathlib.Path,
    ):
        super().__init__(context, paths2topn, topn)
        self.typing_rules = typing_rules

    def extract_predictions_for_file(
        self, path2topn: RichPath, path: pathlib.Path, topn: int
    ) -> RichPath:
        return path2topn

    def annotator(self, annotations: RichPath) -> codemod.Codemod:
        return TypilusFileApplier(
            context=self.context,
            ppath=annotations.path,
            typing_rules=self.typing_rules,
            topn=self.topn,
        )

    def normalisation(self) -> Normalisation:
        return Normalisation(
            typing_text_to_str=True,
            lowercase_aliases=True,
        )


class TypilusFileApplier(codemod.Codemod):
    def __init__(
        self,
        context: codemod.CodemodContext,
        ppath: str,
        typing_rules: pathlib.Path,
        topn: int,
    ) -> None:
        super().__init__(context)
        self.annotator = annotater.Annotater(
            tc=None,
            ppath=ppath,
            granularity="var",
            typing_rules=typing_rules,
        )
        self.topn = topn

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        if (
            new_fpath := self.annotator.annotate(
                fpath=self.context.filename,
                pred_idx=-1,
                type_idx=self.topn,
            )
        ) and new_fpath != self.context.filename:
            try:
                return libcst.parse_module(pathlib.Path(new_fpath).read_text())
            except ParserSyntaxError as e:
                print(pathlib.Path(new_fpath).read_text())
                raise e

        return tree
