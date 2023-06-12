import pathlib
import typing

import libcst
from libcst import codemod, metadata
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

        wrapper = metadata.MetadataWrapper(
            module=tree,
            unsafe_skip_copy=True,
            cache=self.context.metadata_manager.get_cache_for_path(path=self.context.filename),
        )

        try:
            return wrapper.visit(TypeApplier(f_processeed_dict=self.predictions, apply_nlp=False))
        except TypeError:
            # Catch libsa4py bug, return untransformed
            return tree
