import libcst
from libcst import matchers as m, codemod, metadata
from typing import Union

from .base import TaskPreprocessor
from scripts.common.schemas import TypeCollectionCategory

from typet5.experiments import type4py
from typet5.experiments import utils


class StaticPreprocessor(TaskPreprocessor):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        inst_var_rewrite = InstanceVariableWithoutDefaultRewriter(
            context=self.context
        ).transform_module(tree)


class InstanceVariableWithoutDefaultRewriter(codemod.ContextAwareTransformer):
    METADATA_DEPENDENCIES = (metadata.ScopeProvider,)

    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)

    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> libcst.Assign:
        match self.get_metadata(metadata.ScopeProvider, original_node.target):
            case metadata.ClassScope:
                return libcst.Assign(
                    targets=[libcst.AssignTarget(updated_node.target)],
                    value=libcst.Call(updated_node.annotation),
                )
            
            case _:
                return updated_node
