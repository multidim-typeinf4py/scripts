import typing
from typing import Union

import libcst
from libcst import metadata, codemod as c, matchers as m

from common import transformers as t
from common.metadata.anno4inst import Annotation4InstanceProvider, Lowered, TrackedAnnotation


class LoweringTransformer(t.HintableDeclarationTransformer):
    METADATA_DEPENDENCIES = (
        Annotation4InstanceProvider,
        metadata.ParentNodeProvider,
    )

    def __init__(self, context: c.CodemodContext):
        super().__init__(context)

    def _handle_lowering(
        self, _: libcst.CSTNode, unannotated: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        meta: TrackedAnnotation = self.get_metadata(Annotation4InstanceProvider, unannotated)
        if meta.lowered is Lowered.ALTERED:
            assert (lowered_anno := meta.labelled or meta.inferred)
            lowerable_hint = self.get_metadata(metadata.ParentNodeProvider, lowered_anno)
            assert isinstance(lowerable_hint, libcst.AnnAssign)

            lowered_hint = libcst.AnnAssign(
                target=unannotated,
                annotation=lowered_anno,
                value=libcst.Name("λ__LOWERED_HINT_MARKER__λ"),
            )

            return t.Actions((t.Prepend(lowered_hint),))

        return t.Actions((t.Untouched(),))

    def instance_attribute_hint(self, _1: libcst.AnnAssign, _2: libcst.Name) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def libsa4py_hint(self, _1: libcst.Assign, _2: libcst.Name) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def global_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def nonlocal_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def annotated_assignment(
        self, _1: libcst.AnnAssign, _2: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def annotated_hint(self, _1: libcst.AnnAssign, _2: Union[libcst.Name, libcst.Attribute]) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def unannotated_assign_single_target(
        self,
        original_node: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._handle_lowering(original_node, target)

    def unannotated_assign_multiple_targets_or_augassign(
        self,
        original_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._handle_lowering(original_node, target)

    def for_target(
        self, original_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return self._handle_lowering(original_node, target)

    def withitem_target(
        self, original_node: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return self._handle_lowering(original_node, target)


class UnloweringTransformer(c.ContextAwareTransformer):
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> Union[libcst.BaseSmallStatement, libcst.RemovalSentinel]:
        if self.matches(original_node, m.AnnAssign(value=m.Name("λ__LOWERED_HINT_MARKER__λ"))):
            return libcst.RemoveFromParent()
        return updated_node
