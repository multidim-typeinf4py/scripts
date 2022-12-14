import abc
import pathlib

from common.schemas import InferredSchema, InferredSchemaColumns, TypeCollectionSchema

import pandera.typing as pt
import pandas as pd


class Inference(abc.ABC):
    def __init__(self, project: pathlib.Path) -> None:
        super().__init__()
        self.project = project
        self.inferred: pt.DataFrame[InferredSchema] = pt.DataFrame[InferredSchema](
            columns=InferredSchemaColumns
        )

    @abc.abstractmethod
    def infer(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def method(self) -> str:
        pass


class ProjectWideInference(Inference):
    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)

    def infer(self) -> None:
        if self.inferred.empty:
            proj_inf = self._infer_project()
            self.inferred = proj_inf.assign(method=self.method).pipe(pt.DataFrame[InferredSchema])

    @abc.abstractmethod
    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        pass


class PerFileInference(Inference):
    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)

    def infer(self) -> None:
        for subfile in self.project.rglob("*.py"):
            relative = subfile.relative_to(self.project)
            if str(relative) not in self.inferred["file"]:
                reldf = (
                    self._infer_file(relative)
                    .assign(method=self.method)
                    .pipe(pt.DataFrame[InferredSchema])
                )
                self.inferred = (
                    pd.concat([self.inferred, reldf], ignore_index=True)
                    .drop_duplicates(keep="first", ignore_index=True)
                    .pipe(pt.DataFrame[InferredSchema])
                )

    @abc.abstractmethod
    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        pass
