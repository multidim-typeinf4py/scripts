from __future__ import annotations

import dataclasses

import libcst
from libcst import metadata

from common import visitors as v
from common._traversal import T


@dataclasses.dataclass
class TrackedAnnotation:
    annotation: libcst.Annotation
    lowered: bool


class _Annotation4InstanceVisitor(v.HintableDeclarationVisitor, v.ScopeAwareVisitor):
    def __init__(self, provider: Annotation4InstanceProvider):
        super().__init__()
        self.provider = provider

        # Hints provided outside the "scopes" of ifs and elses
        # e.g.
        #
        # a: int | None <---
        # if cond:
        #   a = 5
        # else:
        #   a = None
        self._outer_hinting: list[dict[str, libcst.Annotation]] = []

        # Hints within the active scope
        self._scope_local_hinting: dict[str, libcst.Annotation] = {}

    def visit_If_body(self, _: libcst.If) -> None:
        self._visit_flow_diverging_body(_)

    def visit_Else_body(self, _: libcst.Else) -> None:
        self._visit_flow_diverging_body(_)

    def visit_For_body(self, _: libcst.For) -> None:
        self._visit_flow_diverging_body(_)

    def visit_While_body(self, _: libcst.While) -> None:
        self._visit_flow_diverging_body(_)

    def visit_Try_body(self, _: libcst.Try) -> None:
        self._visit_flow_diverging_body(_)

    def visit_TryStar_body(self, _: libcst.TryStar) -> None:
        self._visit_flow_diverging_body(_)

    def visit_ExceptStarHandler_body(self, _: libcst.ExceptStarHandler) -> None:
        self._visit_flow_diverging_body(_)

    def _visit_flow_diverging_body(self, _: libcst.CSTNode):
        self._outer_hinting.append(self._scope_local_hinting.copy())
        self._scope_local_hinting.clear()

    def leave_If_body(self, _: libcst.If) -> None:
        self._leave_flow_diverging_body(_)

    def leave_Else_body(self, _: libcst.Else) -> None:
        self._leave_flow_diverging_body(_)

    def leave_For_body(self, _: libcst.For) -> None:
        self._leave_flow_diverging_body(_)

    def leave_While_body(self, _: libcst.While) -> None:
        self._leave_flow_diverging_body(_)

    def leave_Try_body(self, _: libcst.Try) -> None:
        self._leave_flow_diverging_body(_)

    def leave_TryStar_body(self, _: libcst.TryStar) -> None:
        self._leave_flow_diverging_body(_)

    def leave_ExceptStarHandler_body(self, _: libcst.CSTNode) -> None:
        self._leave_flow_diverging_body(_)

    def _leave_flow_diverging_body(self, _: libcst.CSTNode):
        self._scope_local_hinting = self._outer_hinting.pop()

    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> None:
        self.provider.set_metadata(
            target, TrackedAnnotation(original_node.annotation, lowered=False)
        )

    def libsa4py_hint(self, _: libcst.Assign, target: libcst.Name) -> None:
        self.provider.set_metadata(target, None)

    def annotated_assignment(
        self, original_node: libcst.AnnAssign, target: libcst.Name | libcst.Attribute
    ) -> None:
        tracked = TrackedAnnotation(original_node.annotation, lowered=False)
        self.provider.set_metadata(target, tracked)
        self._scope_local_hinting[
            self.qualified_name(target)
        ] = original_node.annotation

    def annotated_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name | libcst.Attribute
    ) -> None:
        self._scope_local_hinting[
            self.qualified_name(target)
        ] = original_node.annotation

    def unannotated_assign_single_target(
        self,
        original_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> None:
        annotation = self._retrieve_annotation(target)
        self.provider.set_metadata(target, annotation)

    def unannotated_assign_multiple_targets(
        self,
        original_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def for_target(
        self, original_node: libcst.For, target: libcst.Name | libcst.Attribute
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def withitem_target(
        self, original_node: libcst.With, target: libcst.Name | libcst.Attribute
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def _handle_only_indirectly_annotatable(
        self, target: libcst.Name | libcst.Attribute
    ):
        annotation = self._retrieve_annotation(target)
        self.provider.set_metadata(target, annotation)

    def global_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> None:
        pass

    def nonlocal_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> None:
        pass

    def _retrieve_annotation(
        self, target: libcst.Name | libcst.Attribute
    ) -> TrackedAnnotation | None:
        qname = self.qualified_name(target)
        if (a := self._scope_local_hinting.get(qname, None)) is not None:
            return TrackedAnnotation(annotation=a, lowered=False)

        for hints in reversed(self._outer_hinting):
            if qname in hints:
                return TrackedAnnotation(annotation=hints[qname], lowered=True)
        else:
            return None


class Annotation4InstanceProvider(
    metadata.BatchableMetadataProvider[TrackedAnnotation | None]
):
    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def visit_Module(self, node: libcst.Module) -> None:
        metadata.MetadataWrapper(
            node, unsafe_skip_copy=True, cache=self.metadata
        ).visit(_Annotation4InstanceVisitor(self))
