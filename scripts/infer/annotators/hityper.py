import pathlib
import typing

import libcst
from libcst import codemod, metadata
from libcst.codemod.visitors._apply_type_annotations import Annotations

from scripts.common import ApplyTypeAnnotationsVisitor
from scripts.infer.annotators import ParallelTopNAnnotator
from scripts.infer.annotators.normalisation import Normalisation


class HiTyperProjectApplier(
    ParallelTopNAnnotator[typing.Mapping[pathlib.Path, list[Annotations]], Annotations]
):
    def extract_predictions_for_file(
        self,
        path2topn: typing.Mapping[pathlib.Path, list[Annotations]],
        path: pathlib.Path,
        topn: int,
    ) -> Annotations:
        topn = path2topn[path]
        predictions = topn[self.topn]

        return predictions

    def annotator(self, annotations: Annotations) -> codemod.Codemod:
        return HiTyperFileApplier(context=self.context, annotations=annotations)

    def normalisation(self) -> Normalisation:
        return Normalisation(
            bad_list_generics=True,
            bad_tuple_generics=True,
            bad_dict_generics=True,
            lowercase_aliases=True,
            unnest_union_t=True,
            typing_text_to_str=True,
        )


class HiTyperFileApplier(codemod.Codemod):
    def __init__(
        self, context: codemod.CodemodContext, annotations: Annotations
    ) -> None:
        super().__init__(context)
        self.annotations = annotations

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        return metadata.MetadataWrapper(tree, unsafe_skip_copy=True).visit(
            ApplyTypeAnnotationsVisitor(
                self.context,
                annotations=self.annotations,
            )
        )
