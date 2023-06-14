from libcst import codemod

from scripts.common.schemas import TypeCollectionCategory

class TaskPreprocessor(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, task: TypeCollectionCategory) -> None:
        super().__init__(context)
        self.task = task