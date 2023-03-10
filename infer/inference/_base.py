import abc
import pathlib

from common.schemas import (
    InferredSchema,
    InferredSchemaColumns,
    TypeCollectionSchema,
)

import logging

import pandera.typing as pt
import pandas as pd


class Inference(abc.ABC):

    project: pathlib.Path
    inferred: pt.DataFrame[InferredSchema]

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__()
        self.project = project.resolve()
        self.inferred = InferredSchema.to_schema().example(size=0)

        self.logger = logging.getLogger(type(self).__qualname__)

    @abc.abstractmethod
    def infer(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def method(self) -> str:
        pass


class ProjectWideInference(Inference):
    def infer(self) -> None:
        if self.inferred.empty:
            self.logger.debug(f"Inferring project-wide on {self.project}")
            proj_inf = self._infer_project()
            self.inferred = (
                proj_inf.assign(method=self.method)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

    @abc.abstractmethod
    def _infer_project(self) -> pt.DataFrame[InferredSchema]:
        pass


class PerFileInference(Inference):
    def infer(self) -> None:
        updates = list()
        for subfile in self.project.rglob("*.py"):
            relative = subfile.relative_to(self.project)
            if str(relative) not in self.inferred["file"]:
                self.logger.debug(f"Inferring per-file on {self.project} @ {relative}")
                reldf: pt.DataFrame[InferredSchema] = self._infer_file(relative)
                updates.append(reldf)
        if updates:
            self.inferred = (
                pd.concat([self.inferred, *updates], ignore_index=True)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

    @abc.abstractmethod
    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        pass
