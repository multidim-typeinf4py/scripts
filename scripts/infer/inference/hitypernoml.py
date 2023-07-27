import pathlib

import libcst
from libcst import codemod

from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial
from ...common.schemas import TypeCollectionCategory


class NoOpPreprocessor(codemod.Codemod):
    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        return tree

class NoMLAdaptor(ModelAdaptor):
    def topn(self) -> int:
        return 0

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        return ModelAdaptor.ProjectPredictions(__root__=dict())

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return NoOpPreprocessor(context=codemod.CodemodContext())


class HiTyperNoML(HiTyper):
    def __init__(self):
        super().__init__(NoMLAdaptor())

    def method(self) -> str:
        return f"HiTyperNoML"
