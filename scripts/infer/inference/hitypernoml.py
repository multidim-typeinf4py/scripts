import pathlib

import libcst
from libcst import codemod

from scripts.infer.preprocessers.tt5 import TT5Preprocessor

from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial
from scripts.common.schemas import TypeCollectionCategory

class NoOpPreprocessor(codemod.ContextAwareTransformer):
    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.FunctionDef:
        return updated_node.with_changes(returns=None)

    def leave_Param(self, original_node: libcst.Param, updated_node: libcst.Param) -> libcst.Param:
        return updated_node.with_changes(annotation=None)

class NoMLAdaptor(ModelAdaptor):
    def topn(self) -> int:
        return 1

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        return ModelAdaptor.ProjectPredictions(__root__=dict())

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return TT5Preprocessor(context=codemod.CodemodContext(), task="all")


class HiTyperNoML(HiTyper):
    def __init__(self):
        super().__init__(NoMLAdaptor())

    def method(self) -> str:
        return f"HiTyperNoML"
