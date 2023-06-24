import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


from scripts.infer.inference.t4py import Type4PyTopN
from . import _utils
from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial

from libcst import codemod

from scripts.common.output import InferenceArtifactIO
from scripts.common.schemas import TypeCollectionCategory


class Type4PyAdaptor(ModelAdaptor):
    def __init__(
        self,
        topn: int,
    ) -> None:
        super().__init__()
        self.topn = topn

    def topn(self) -> int:
        return self.topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        io = InferenceArtifactIO(
            artifact_root=pathlib.Path(os.environ.get("ARTIFACT_ROOT")),
            dataset=os.environ.get("DATASET_STRUCTURE"),
            repository=pathlib.Path(os.environ.get("REPOSITORY")),
        )

        (type4py_predictions,) = io.read()

        # Make relative to temporary project root
        root: dict[str, ModelAdaptor.FilePredictions] = {
            str(project.resolve() / file): ModelAdaptor.FilePredictions.parse_obj(p)
            for file, p in type4py_predictions.items()
            if p
        }
        return ModelAdaptor.ProjectPredictions(__root__=root)

    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        return self.type4py.preprocessor(task=task)


class HiType4PyTopN(HiTyper):
    def __init__(
        self,
        topn: int,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__(
            Type4PyAdaptor(
                topn=topn,
                cpu_executor=cpu_executor,
                model_executor=model_executor,
            )
        )

    def method(self) -> str:
        return f"HiType4PyN{self.adaptor.topn()}"


HiType4PyTop1 = wrapped_partial(HiType4PyTopN, topn=1)
HiType4PyTop3 = wrapped_partial(HiType4PyTopN, topn=3)
HiType4PyTop5 = wrapped_partial(HiType4PyTopN, topn=5)
HiType4PyTop10 = wrapped_partial(HiType4PyTopN, topn=10)
