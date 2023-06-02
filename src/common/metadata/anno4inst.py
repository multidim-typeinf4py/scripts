from __future__ import annotations

import dataclasses
import enum
from typing import Optional, Union

import libcst
from libcst import metadata

from src.common import visitors as v


class Lowered(enum.Enum):
    UNALTERED = enum.auto()
    ALTERED = enum.auto()


@dataclasses.dataclass
class TrackedAnnotation:
    labelled: Optional[libcst.Annotation]
    inferred: Optional[libcst.Annotation]
    lowered: Lowered


class _Consumption(enum.Enum):
    UNUSED = enum.auto()
    USED = enum.auto()


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
        self._outer_hinting: list[
            dict[str, tuple[libcst.Annotation, _Consumption, Lowered]]
        ] = []

        # Hints within the active scope
        self._scope_local_hinting: dict[
            str, tuple[libcst.Annotation, _Consumption, Lowered]
        ] = {}

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

    def leave_ExceptStarHandler_body(self, _: libcst.ExceptStarHandler) -> None:
        self._leave_flow_diverging_body(_)

    def _visit_flow_diverging_body(self, _: libcst.CSTNode):
        self._outer_hinting.append(self._scope_local_hinting.copy())
        self._scope_local_hinting.clear()

    def _leave_flow_diverging_body(self, _: libcst.CSTNode):
        self._scope_local_hinting = self._outer_hinting.pop()

    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> None:
        meta = TrackedAnnotation(
            labelled=original_node.annotation,
            inferred=original_node.annotation,
            lowered=Lowered.UNALTERED,
        )
        self.provider.set_metadata(target, meta)

    def libsa4py_hint(self, node: Union[libcst.AnnAssign, libcst.Assign], target: libcst.Name) -> None:
        annotation = node.annotation if isinstance(node, libcst.AnnAssign) else None
        self.provider.set_metadata(
            target,
            TrackedAnnotation(labelled=annotation, inferred=None, lowered=Lowered.UNALTERED),
        )

    def annotated_assignment(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        meta = TrackedAnnotation(
            labelled=original_node.annotation,
            inferred=original_node.annotation,
            lowered=Lowered.UNALTERED,
        )
        self.provider.set_metadata(target, meta)
        self._track_annotation(
            qname=self.qualified_name(target),
            consumption=_Consumption.USED,
            lowerage=Lowered.UNALTERED,
            annotation=original_node.annotation,
        )

    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._track_annotation(
            qname=self.qualified_name(target),
            consumption=_Consumption.UNUSED,
            lowerage=Lowered.UNALTERED,
            annotation=original_node.annotation,
        )

    def assign_single_target(
        self,
        original_node: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        if tracked := self._retrieve_for_unannotated(target):
            annotation, _, lowered = tracked

            # Always treat Assign as inferred and not labelled due to not being an AnnAssign
            self.provider.set_metadata(
                target,
                TrackedAnnotation(labelled=None, inferred=annotation, lowered=lowered),
            )

        else:
            self.provider.set_metadata(
                target,
                TrackedAnnotation(
                    labelled=None, inferred=None, lowered=Lowered.UNALTERED
                ),
            )

    def assign_multiple_targets_or_augassign(
        self,
        original_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def for_target(
        self, original_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def withitem_target(
        self, original_node: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._handle_only_indirectly_annotatable(target)

    def _handle_only_indirectly_annotatable(
        self, target: Union[libcst.Name, libcst.Attribute]
    ):
        if tracked := self._retrieve_for_unannotated(target):
            annotation, consumption, lowered = tracked
            if consumption is _Consumption.UNUSED:
                self.provider.set_metadata(
                    target,
                    TrackedAnnotation(
                        labelled=annotation, inferred=annotation, lowered=lowered
                    ),
                )
            else:
                self.provider.set_metadata(
                    target,
                    TrackedAnnotation(
                        labelled=None, inferred=annotation, lowered=lowered
                    ),
                )
        else:
            self.provider.set_metadata(
                target,
                TrackedAnnotation(
                    labelled=None, inferred=None, lowered=Lowered.UNALTERED
                ),
            )

    def global_target(
        self,
        original_node: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        pass

    def nonlocal_target(
        self,
        original_node: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        pass

    def _retrieve_for_unannotated(
        self, target: Union[libcst.Name, libcst.Attribute]
    ) -> tuple[libcst.Annotation, _Consumption, Lowered] | None:
        result: tuple[libcst.Annotation, _Consumption, Lowered] | None = None
        qname = self.qualified_name(target)

        if a := self._scope_local_hinting.get(qname, None):
            # Replace entry with USED, retain lowerage status
            anno, _, lowered = result = a
            self._track_annotation(qname, _Consumption.USED, lowered, annotation=anno)

        elif hints := next(
            filter(lambda h: qname in h, reversed(self._outer_hinting)),
            None,
        ):
            # Add entry to local hinting as ALTERED
            anno, _, _ = hints[qname]
            result = (anno, _Consumption.USED, Lowered.ALTERED)
            self._track_annotation(
                qname, _Consumption.USED, Lowered.ALTERED, annotation=anno
            )

        return result

    def _track_annotation(
        self,
        qname: str,
        consumption: _Consumption,
        lowerage: Lowered,
        annotation: libcst.Annotation,
    ) -> libcst.Annotation:
        self._scope_local_hinting[qname] = (annotation, consumption, lowerage)

        # Insert preprocessing here if wished

        return annotation


class Annotation4InstanceProvider(
    metadata.BatchableMetadataProvider[TrackedAnnotation]
):
    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def visit_Module(self, node: libcst.Module) -> None:
        metadata.MetadataWrapper(
            node, unsafe_skip_copy=True, cache=self.metadata
        ).visit(_Annotation4InstanceVisitor(self))
