import os
import pathlib

from libcst import codemod

from scripts.common.schemas import TypeCollectionCategory
from scripts.infer.preprocessers.t4py import Type4PyPreprocessor
from scripts.infer.structure import DatasetFolderStructure
from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial


class Type4PyAdaptor(ModelAdaptor):
    def __init__(
        self,
        topn: int,
    ) -> None:
        super().__init__()
        self._topn = topn

    def topn(self) -> int:
        return self._topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        from scripts.common.output import InferenceArtifactIO

        io = InferenceArtifactIO(
            artifact_root=pathlib.Path(os.environ["ARTIFACT_ROOT"]),
            dataset=DatasetFolderStructure(pathlib.Path(os.environ["DATASET_ROOT"])),
            repository=pathlib.Path(os.environ["REPOSITORY"]),
            tool_name=f"type4pyN{self.topn()}",
            task=os.environ["TASK"],
        )

        # No need to trim predictions; HiTyper does this for us via
        # tdg.recommendType(self, ..., topn)
        (type4py_predictions,) = io.read()

        # Make relative to temporary project root
        root: dict[str, ModelAdaptor.FilePredictions] = {
            str(project.resolve() / file): ModelAdaptor.FilePredictions.parse_obj(p)
            for file, p in type4py_predictions.items()
            if p
        }

        return ModelAdaptor.ProjectPredictions(__root__=root)

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return Type4PyPreprocessor(context=codemod.CodemodContext(), task=task)


class HiType4PyTopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(Type4PyAdaptor(topn=topn))

    def method(self) -> str:
        return f"HiType4PyN{self.adaptor.topn()}"


HiType4PyTop1 = wrapped_partial(HiType4PyTopN, topn=1)
HiType4PyTop3 = wrapped_partial(HiType4PyTopN, topn=3)
HiType4PyTop5 = wrapped_partial(HiType4PyTopN, topn=5)
HiType4PyTop10 = wrapped_partial(HiType4PyTopN, topn=10)
