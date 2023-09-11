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
        artifact_dir: pathlib.Path,
    ):
        super().__init__(context, paths2topn, topn)
        self.typing_rules = typing_rules
        self.artifact_dir = artifact_dir

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
            artifact_out=self.artifact_dir,
        )

    def normalisation(self) -> Normalisation:
        return Normalisation.default()


class TypilusFileApplier(codemod.Codemod):
    def __init__(
        self,
        context: codemod.CodemodContext,
        ppath: str,
        typing_rules: pathlib.Path,
        topn: int,
        artifact_out: pathlib.Path,
    ) -> None:
        super().__init__(context)
        self.annotator = annotater.Annotater(
            tc=None,
            ppath=ppath,
            granularity="var",
            typing_rules=typing_rules,
        )
        self.topn = topn
        self.artifact_out = artifact_out

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        # try:
        new_fpath = self.annotator.annotate(
            fpath=self.context.filename,
            pred_idx=-1,
            type_idx=self.topn,
        )
        if new_fpath != self.context.filename:
            code = pathlib.Path(new_fpath).read_text()
            try:
                return libcst.parse_module(code)

            except Exception:
                print(self.context.filename, code)

            finally:
                import json, os

                artifact_out = (
                    self.artifact_out / self.context.filename.replace(os.sep, "-")
                ).with_suffix(".artifact")
                assert not artifact_out.is_file()

                # print(self.context.filename, self.annotator.prediction_with_quals)
                with artifact_out.open("w") as f:
                    # print(
                    #    self.context.filename,
                    #    "->",
                    #    artifact_out,
                    #    "->",
                    #    self.annotator.prediction_with_quals[:3],
                    # )
                    relfname = pathlib.Path(self.context.filename).relative_to(
                        self.context.metadata_manager.root_path
                    )
                    json.dump(
                        {str(relfname): self.annotator.prediction_with_quals},
                        f,
                        indent=2,
                    )

        return tree
        #    code = tree.code

        # except Exception as e:
        #    print(code)
        #    raise e
