import libcst
from libcst import matchers as m, codemod
from typing import Union

from .base import TaskPreprocessor, AnnotationRemover
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import type4py
from typet5.experiments import utils


class Type4PyPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        # simpler_syntax = utils.remove_newer_syntax(tree, supported=type4py.Type4PySupportedSyntax)
        return Type4PyAnnotationRemover(
            context=self.context,
            task=self.task,
        ).transform_module(tree)


class Type4PyAnnotationRemover(AnnotationRemover):
    # Adapted from libsa4PY
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> Union[libcst.BaseSmallStatement, libcst.RemovalSentinel]:
        if self.task is not TypeCollectionCategory.VARIABLE:
            return updated_node
        # It handles a special case where a type-annotated variable has not initialized, e.g. foo: str
        # This case will be converted to foo = ... so that nodes traversal won't encounter exceptions later on
        if m.matches(
            original_node,
            m.AnnAssign(
                target=m.Name(value=m.DoNotCare()),
                annotation=m.Annotation(annotation=m.DoNotCare()),
                value=None,
            ),
        ):
            updated_node = libcst.Assign(
                targets=[libcst.AssignTarget(target=original_node.target)], value=libcst.Ellipsis()
            )
        # Handles type-annotated class attributes that has not been initialized, e.g. self.foo: str
        elif m.matches(
            original_node,
            m.AnnAssign(
                target=m.Attribute(value=m.DoNotCare()),
                annotation=m.Annotation(annotation=m.DoNotCare()),
                value=None,
            ),
        ):
            updated_node = libcst.Assign(
                targets=[libcst.AssignTarget(target=original_node.target)], value=libcst.Ellipsis()
            )
        else:
            updated_node = libcst.Assign(
                targets=[libcst.AssignTarget(target=original_node.target)],
                value=original_node.value,
            )
        return updated_node
