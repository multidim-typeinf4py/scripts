import pathlib
import pprint
import typing

import libcst
from libcst import codemod, metadata, matchers as m, MaybeSentinel
from libsa4py.cst_transformers import TypeApplier

from scripts.infer.annotators import ParallelTopNAnnotator
from .normalisation import Normalisation


class Type4PyProjectApplier(ParallelTopNAnnotator[typing.Mapping[pathlib.Path, list[dict]], dict]):
    def extract_predictions_for_file(
        self,
        path2topn: typing.Mapping[pathlib.Path, list[dict]],
        path: pathlib.Path,
        topn: int,
    ) -> dict:
        if path not in path2topn:
            return dict()
        topn_predictions = path2topn[path]
        predictions = topn_predictions[topn]

        return predictions

    def annotator(self, annotations: dict) -> codemod.Codemod:
        return Type4PyFileApplier(context=self.context, predictions=annotations)

    def normalisation(self) -> Normalisation:
        return Normalisation.default()


class Type4PyFileApplier(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, predictions: dict) -> None:
        super().__init__(context=context)
        self.predictions = predictions

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        if self.predictions and (
            self.predictions.get("classes")
            or self.predictions.get("funcs")
            or self.predictions.get("variables")
        ):
            try:
                tree = metadata.MetadataWrapper(
                    module=tree,
                    unsafe_skip_copy=True,
                ).visit(TypeApplier(f_processeed_dict=self.predictions, apply_nlp=False))
            except Exception as e:
                print(f"Failed to annotate {self.context.filename}! Predictions were:")
                pprint.pprint(self.predictions)
                print(e)

                import sys, traceback

                traceback.print_exc()

        # Always remove artifacts, even if annotation process was unsuccessful
        #without_libsa4py_artifacts = RemoveLibSa4PyArtifacts(context=self.context).transform_module(
        #    tree
        #)

        return tree


class RemoveLibSa4PyArtifacts(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.AnnAssign(value=m.Ellipsis()))
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.AnnAssign:
        return updated_node.with_changes(value=None, equal=MaybeSentinel.DEFAULT)

    # This will not nuke old assignments in the original source code matching
    # $NAME = ...
    # as we transform them into
    # $NAME = None
    # in the preprocessor
    @m.call_if_inside(m.Assign(value=m.Ellipsis()))
    def leave_Assign(
        self, original_node: libcst.Assign, updated_node: libcst.Assign
    ) -> libcst.RemovalSentinel:
        return libcst.RemoveFromParent()

    # NOTE: libcst.Assigns with m.matches(updated_node.value, m.Ellipsis())
    # NOTE: should not occur, but if they do, they are a libsa4py bug
    # NOTE: Check datapoints in dataset and collected from libsa4py are identical
