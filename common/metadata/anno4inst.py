from __future__ import annotations

import dataclasses

import libcst
from libcst import metadata

from common import visitors as v


@dataclasses.dataclass
class TrackedAnnotation:
    annotation: libcst.Annotation
    hoisted: bool


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
        self, target: libcst.Name, annotation: libcst.Annotation | None
    ) -> None:
        if annotation is not None:
            self.provider.set_metadata(target, TrackedAnnotation(annotation, hoisted=False))
        else:
            self.provider.set_metadata(target, None)

    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        md = TrackedAnnotation(annotation, hoisted=False)
        self.provider.set_metadata(target, md)

        if isinstance(target, libcst.Attribute):
            self.provider.set_metadata(target.value, md)

        self._scope_local_hinting[self.qualified_name(target)] = annotation

    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        self._scope_local_hinting[self.qualified_name(target)] = annotation

    def unannotated_target(self, target: libcst.Name | libcst.Attribute) -> None:
        annotation = self._retrieve_annotation(target)

        self.provider.set_metadata(target, annotation)
        if isinstance(target, libcst.Attribute):
            self.provider.set_metadata(target.value, annotation)

    def _retrieve_annotation(
        self, target: libcst.Name | libcst.Attribute
    ) -> TrackedAnnotation | None:
        qname = self.qualified_name(target)
        if (a := self._scope_local_hinting.get(qname, None)) is not None:
            return TrackedAnnotation(annotation=a, hoisted=False)

        for hints in reversed(self._outer_hinting):
            if qname in hints:
                return TrackedAnnotation(annotation=hints[qname], hoisted=True)
        else:
            return None

    def scope_overwritten_target(self, _: libcst.Name) -> None:
        ...


class Annotation4InstanceProvider(metadata.BatchableMetadataProvider[TrackedAnnotation | None]):
    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def visit_Module(self, node: libcst.Module) -> None:
        metadata.MetadataWrapper(node, unsafe_skip_copy=True, cache=self.metadata).visit(
            _Annotation4InstanceVisitor(self)
        )
