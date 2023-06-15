import libcst
from libcst import matchers as m, codemod, metadata
from typing import Union

from .base import TaskPreprocessor, AnnotationRemover
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import type4py
from typet5.experiments import utils


class StaticPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        rewritten = _StaticAnnotationRemover(
            context=self.context,
            task=self.task,
        ).transform_module(tree)
        return rewritten


class _StaticAnnotationRemover(AnnotationRemover):
    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.Assign | libcst.AnnAssign | libcst.RemovalSentinel:
        if self.task is not TypeCollectionCategory.VARIABLE:
            return updated_node

        is_class_scope = self.is_class_scope(original_node.target)
        if is_class_scope and original_node.target is None:
            # This is similar to TypeT5's published implementation, which makes a = int
            # a: int -> a = int()
            return libcst.Assign(
                targets=[libcst.AssignTarget(updated_node.target)],
                value=libcst.Call(updated_node.annotation.annotation),
            )

        elif not is_class_scope and original_node.target is None:
            # a: int -> removed
            return libcst.RemoveFromParent()

        else:
            # a: int = 5 -> a = 5
            # both inside of and outside of classes
            return libcst.Assign(
                targets=[libcst.AssignTarget(updated_node.target)],
                value=updated_node.value
            )
