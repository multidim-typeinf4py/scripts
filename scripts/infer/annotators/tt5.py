import pathlib
import typing

import libcst
import mpmath
from libcst import codemod, matchers as m

from typet5.static_analysis import SignatureMap, SignatureMapTopN
from typet5.experiments import utils as typet5_utils

from .tool_annotator import ParallelTopNAnnotator, U, T


class TT5ProjectApplier(ParallelTopNAnnotator[SignatureMapTopN, SignatureMap]):
    def extract_predictions_for_file(
        self, path2topn: SignatureMapTopN, path: pathlib.Path, topn: int
    ) -> SignatureMap:
        return SignatureMap(
            (project_path, signatures[topn])
            for project_path, signatures in path2topn.items()
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
        normalised = TT5FileNormalizer(context=self.context).transform_module(
            typet5_annotated
        )

        return normalised


class TT5FileNormalizer(codemod.ContextAwareTransformer):
    @m.call_if_inside(m.Annotation())
    def leave_Tuple(
        self, original_node: libcst.Tuple, updated_node: libcst.Tuple
    ) -> libcst.BaseExpression:
        if len(updated_node.elements) == 0:
            return libcst.Name("Tuple")

        return libcst.Subscript(
            value=libcst.Name("Tuple"),
            slice=self.replace_elements(updated_node.elements),
        )

    @m.call_if_inside(m.Annotation())
    def leave_List(
        self, original_node: libcst.List, updated_node: libcst.List
    ) -> libcst.BaseExpression:
        if len(updated_node.elements) == 0:
            return libcst.Name("List")

        elif len(updated_node.elements) > 1:
            list_typing = [
                libcst.SubscriptElement(
                    libcst.Index(
                        libcst.Subscript(
                            value=libcst.Name("Union"),
                            slice=self.replace_elements(updated_node.elements),
                        )
                    )
                )
            ]

        else:
            list_typing = self.replace_elements(updated_node.elements)

        return libcst.Subscript(
            value=libcst.Name("List"),
            slice=list_typing,
        )

    def replace_elements(
        self, elements: typing.Sequence[libcst.BaseElement]
    ) -> list[libcst.SubscriptElement]:
        assert all(map(lambda e: isinstance(e, libcst.Element), elements))
        return [libcst.SubscriptElement(libcst.Index(value=e.value)) for e in elements]
