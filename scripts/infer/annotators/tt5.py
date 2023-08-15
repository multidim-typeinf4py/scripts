import pathlib

import libcst
from libcst import codemod
from typet5.experiments import utils as typet5_utils
from typet5.function_decoding import RolloutPrediction
from typet5.static_analysis import SignatureMap

from .normalisation import Normalisation
from .tool_annotator import ParallelTopNAnnotator


class TT5ProjectApplier(ParallelTopNAnnotator[RolloutPrediction, SignatureMap]):
    def extract_predictions_for_file(
        self, path2topn: RolloutPrediction, path: pathlib.Path, topn: int
    ) -> SignatureMap:
        return path2topn.predicted_sigmap

    def annotator(self, annotations: SignatureMap) -> codemod.Codemod:
        return TT5FileApplier(context=self.context, sigmap=annotations)

    def normalisation(self) -> Normalisation:
        return Normalisation.default()


class TT5FileApplier(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, sigmap: SignatureMap) -> None:
        super().__init__(context=context)
        self.sigmap = sigmap

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        typet5_annotated = typet5_utils.apply_sigmap(
            m=tree,
            sigmap=self.sigmap,
            module_name=self.context.full_module_name,
        )

        return typet5_annotated
