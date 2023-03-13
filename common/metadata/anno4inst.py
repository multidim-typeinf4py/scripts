from __future__ import annotations

import libcst
from libcst import metadata

from common import visitors as v


class _Annotation4InstanceVisitor(v.HintableDeclarationVisitor, v.ScopeAwareVisitor):
    def __init__(self, provider: Annotation4InstanceProvider):
        super().__init__()
        self.provider = provider

        self._hinting: dict[str, libcst.Annotation] = {}

    def instance_attribute_hint(
        self, target: libcst.Name, annotation: libcst.Annotation | None
    ) -> None:
        self.provider.set_metadata(target, annotation)

    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        self.provider.set_metadata(target, annotation)

        if isinstance(target, libcst.Attribute):
            self.provider.set_metadata(target.value, annotation)

        self._hinting.pop(self.qualified_name(target), None)

    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        self._hinting[self.qualified_name(target)] = annotation

    def unannotated_target(self, target: libcst.Name | libcst.Attribute) -> None:
        # Consume annotation here!
        annotation = self._hinting.pop(self.qualified_name(target), None)
        self.provider.set_metadata(target, annotation)

        if isinstance(target, libcst.Attribute):
            self.provider.set_metadata(target.value, annotation)

    def scope_overwritten_target(self, _: libcst.Name) -> None:
        ...


class Annotation4InstanceProvider(metadata.BatchableMetadataProvider[libcst.Annotation | None]):
    def visit_Module(self, node: libcst.Module) -> bool | None:
        metadata.MetadataWrapper(node, unsafe_skip_copy=True, cache=self.metadata).visit(
            _Annotation4InstanceVisitor(self)
        )
        return None
