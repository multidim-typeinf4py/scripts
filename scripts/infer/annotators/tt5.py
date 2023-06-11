import pathlib

import libcst
from libcst import codemod

from typet5.static_analysis import SignatureMap, SignatureMapTopN
from typet5.experiments import utils as typet5_utils


from scripts.infer.normalisers import bracket
from .tool_annotator import ParallelTopNAnnotator


class TT5ProjectApplier(ParallelTopNAnnotator[SignatureMapTopN, SignatureMap]):
    def extract_predictions_for_file(
        self, path2topn: SignatureMapTopN, path: pathlib.Path, topn: int
    ) -> SignatureMap:
        return SignatureMap(
            (project_path, signatures[topn]) for project_path, signatures in path2topn.items()
        )

    def annotator(self, annotations: SignatureMap) -> codemod.Codemod:
        return TT5FileApplier(context=self.context, sigmap=annotations)


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

        return (
            typet5_annotated \
            .visit(bracket.RoundBracketsToTuple(context=self.context)) \
            .visit(bracket.SquareBracketsToList(context=self.context))
        )
