import abc
from contextlib import contextmanager
import os
import pathlib
import tempfile
import typing
import shutil

from common.schemas import (
    InferredSchema,
    InferredSchemaColumns,
    TypeCollectionSchema,
    TypeCollectionCategory,
)

import pandera.typing as pt
import pandas as pd


@contextmanager
def scratchpad(untouched: pathlib.Path) -> typing.Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(src=str(untouched), dst=td, dirs_exist_ok=True)
        try:
            yield pathlib.Path(td)
        finally:
            pass


@contextmanager
def working_dir(wd: pathlib.Path) -> typing.Generator[None, None, None]:
    oldcwd = pathlib.Path.cwd()
    os.chdir(wd)

    try:
        yield
    finally:
        os.chdir(oldcwd)


class Inference(abc.ABC):

    project: pathlib.Path
    inferred: pt.DataFrame[InferredSchema]

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__()
        self.project = project.resolve()
        self.inferred = pt.DataFrame[InferredSchema](columns=InferredSchemaColumns)

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
            proj_inf = self._infer_project()
            self.inferred = (
                proj_inf.assign(method=self.method)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

    @abc.abstractmethod
    def _infer_project(self) -> pt.DataFrame[TypeCollectionSchema]:
        pass


class PerFileInference(Inference):
    def infer(self) -> None:
        updates = list()
        for subfile in self.project.rglob("*.py"):
            relative = subfile.relative_to(self.project)
            if str(relative) not in self.inferred["file"]:
                reldf: pt.DataFrame[InferredSchema] = (
                    self._infer_file(relative)
                    .assign(method=self.method)
                    .pipe(pt.DataFrame[InferredSchema])
                )
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
