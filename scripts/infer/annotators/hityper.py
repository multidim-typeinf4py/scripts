import pathlib
import typing

import libcst
from libcst import codemod
from typet5.experiments import utils as typet5_utils
from typet5.static_analysis import SignatureMap

from scripts.infer.annotators import ParallelTopNAnnotator
from scripts.infer.annotators.normalisation import Normalisation


class HiTyperProjectApplier(
    ParallelTopNAnnotator[typing.Mapping[pathlib.Path, list[SignatureMap]], SignatureMap]
):
    def extract_predictions_for_file(
        self,
        path2topn: typing.Mapping[pathlib.Path, list[SignatureMap]],
        path: pathlib.Path,
        topn: int,
    ) -> SignatureMap:
        topns = path2topn.get(path, [{}])
        return topns[topn]


    def annotator(self, sigmap: SignatureMap) -> codemod.Codemod:
        return HiTyperFileApplier(context=self.context, sigmap=sigmap)

    def normalisation(self) -> Normalisation:
        return Normalisation.default()


class HiTyperFileApplier(codemod.Codemod):
    def __init__(
        self, context: codemod.CodemodContext, sigmap: SignatureMap
    ) -> None:
        super().__init__(context)
        self.sigmap = sigmap

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        typet5_annotated = typet5_utils.apply_sigmap(
            m=tree,
            sigmap=self.sigmap,
            module_name=self.context.full_module_name,
        )

        return typet5_annotated