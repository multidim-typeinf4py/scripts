import functools
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from scripts.infer.inference.t4py import Type4PyTopN
from . import _utils
from ._hityper import ModelAdaptor, HiTyper
from ._utils import wrapped_partial

from libcst import codemod
from scripts.common.schemas import TypeCollectionCategory



class Type4PyAdaptor(ModelAdaptor):
    def __init__(
        self,
        topn: int,
        cpu_executor: ProcessPoolExecutor,
        model_executor: ThreadPoolExecutor,
    ) -> None:
        super().__init__()
        self.type4py = Type4PyTopN(
            topn=topn,
            cpu_executor=cpu_executor,
            model_executor=model_executor,
        )

    def topn(self) -> int:
        return self.type4py.topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        datapoints = self.type4py.extract_datapoints(project, subset)
        predictions = self.type4py.make_predictions(datapoints, subset)

        root: dict[str, ModelAdaptor.FilePredictions] = {
            str(project.resolve() / file): ModelAdaptor.FilePredictions.parse_obj(p)
            for file, p in predictions.items()
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
