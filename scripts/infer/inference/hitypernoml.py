import pathlib

from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial

class NoMLAdaptor(ModelAdaptor):
    def topn(self) -> int:
        return 0

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        return ModelAdaptor.ProjectPredictions(__root__=dict())

class HiTyperNoML(HiTyper):
    def __init__(self):
        super().__init__(NoMLAdaptor())

    def method(self) -> str:
        return f"HiTyperNoML"
