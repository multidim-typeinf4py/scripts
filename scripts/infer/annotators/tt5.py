import pathlib

import libcst
from libcst import codemod

from typet5.static_analysis import SignatureMap, SignatureMapTopN
from typet5.experiments import utils as typet5_utils


from .tool_annotator import ParallelTopNAnnotator
from .normalisation import Normalisation


class TT5ProjectApplier(ParallelTopNAnnotator[SignatureMapTopN, SignatureMap]):
    def extract_predictions_for_file(
        self, path2topn: SignatureMapTopN, path: pathlib.Path, topn: int
    ) -> SignatureMap:
        assert self.context.full_module_name is not None
        return SignatureMap(
            (project_path, signatures[topn])
            for project_path, signatures in path2topn.items()
            if project_path.module == self.context.full_module_name
        )

    def annotator(self, annotations: SignatureMap) -> codemod.Codemod:
        return TT5FileApplier(context=self.context, sigmap=annotations)

    def normalisation(self) -> Normalisation:
        return Normalisation(
            bad_generics=True,
            normalise_union_ts=True,
            remove_if_all_any=True,
            lowercase_aliases=True
        )


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
