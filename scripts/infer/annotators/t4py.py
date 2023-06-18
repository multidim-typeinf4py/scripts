import pathlib
import pprint
import typing

import libcst
from libcst import codemod, metadata, matchers as m, MaybeSentinel
from libsa4py.cst_transformers import TypeApplier

from scripts.infer.annotators import ParallelTopNAnnotator
from scripts.infer.annotators.tool_annotator import Normalisation


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
        return Normalisation(
            bad_list_generics=True,
            bad_tuple_generics=True,
            bad_dict_generics=True,
            bad_literals=True,
            typing_text_to_str=True,
            lowercase_aliases=True,
        )


class Type4PyFileApplier(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, predictions: dict) -> None:
        super().__init__(context=context)
        self.predictions = predictions

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        if not self.predictions:
            return tree

        if not (
            self.predictions.get("classes")
            or self.predictions.get("funcs")
            or self.predictions.get("variables")
        ):
            return tree

        wrapper = metadata.MetadataWrapper(
            module=tree,
            unsafe_skip_copy=True,
        )

        try:
            annotated = wrapper.visit(
                TypeApplier(f_processeed_dict=self.predictions, apply_nlp=False)
            )
            without_libsa4py_artifacts = RemoveLibSa4PyArtifacts(
                context=self.context
            ).transform_module(annotated)

            return without_libsa4py_artifacts
        except Exception:
            print(f"Failed to annotate {self.context.filename}! Predictions were:")
            pprint.pprint(self.predictions)
            return tree


class RemoveLibSa4PyArtifacts(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.AnnAssign(value=m.Ellipsis()))
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.AnnAssign:
        return updated_node.with_changes(value=None, equal=MaybeSentinel.DEFAULT)

    # NOTE: libcst.Assigns with m.matches(updated_node.value, m.Ellipsis())
    # NOTE: should not occur, but if they do, they are a libsa4py bug
    # NOTE: Check datapoints in dataset and collected from libsa4py are identical
