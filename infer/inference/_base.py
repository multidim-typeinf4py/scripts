import abc
import enum
import pathlib
import typing

from common.schemas import (
    InferredSchema,
    InferredSchemaColumns,
    TypeCollectionSchema,
)

import logging

import pandera.typing as pt
import pandas as pd


class DatasetFolderStructure(enum.Enum):
    MANYTYPES4PY = enum.auto()
    TYPILUS = enum.auto()

    def project_iter(
        self, dataset_root: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        if self == DatasetFolderStructure.MANYTYPES4PY:
            authors = (
                author
                for author in dataset_root.iterdir()
                if author.is_dir() and not author.name.startswith(".")
            )
            for author in authors:
                repos = (repo for repo in author.iterdir() if repo.is_dir())
                yield from repos

        elif self == DatasetFolderStructure.TYPILUS:
            repos = (
                repo
                for repo in dataset_root.iterdir()
                if repo.is_dir()
                and not repo.name.startswith(".")
                and len(repo.name.split(".")) == 2
            )
            yield from repos


class Inference(abc.ABC):
    dataset: pathlib.Path
    inferred: pt.DataFrame[InferredSchema]

    def __init__(self, dataset: pathlib.Path) -> None:
        super().__init__()
        self.dataset = dataset.resolve()
        self.inferred = InferredSchema.example(size=0)

        self.logger = logging.getLogger(type(self).__qualname__)

    @abc.abstractmethod
    def infer(self, structure: DatasetFolderStructure) -> None:
        pass

    @property
    @abc.abstractmethod
    def method(self) -> str:
        pass


class DatasetWideInference(Inference):
    def __init__(self, dataset: pathlib.Path) -> None:
        super().__init__(dataset)
        self.structure = None

    def infer(self, structure: DatasetFolderStructure) -> None:
        self.inferred = self._infer_dataset(structure)

    @abc.abstractmethod
    def _infer_dataset(
        self, structure: DatasetFolderStructure
    ) -> pt.DataFrame[InferredSchema]:
        ...


class ProjectWideInference(Inference):
    def infer(self, structure: DatasetFolderStructure) -> None:
        for project in structure.project_iter(self.dataset):
            self.logger.debug(f"Inferring project-wide on {project}")
            proj_inf = self._infer_project(project)
            self.inferred = (
                proj_inf.assign(method=self.method)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

    @abc.abstractmethod
    def _infer_project(self, project: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        pass


class PerFileInference(Inference):
    def infer(self, structure: DatasetFolderStructure) -> None:
        updates = list()
        for project in structure.project_iter(self.dataset):
            for subfile in project.rglob("*.py"):
                relative = subfile.relative_to(project)
                if str(relative) not in self.inferred["file"]:
                    self.logger.debug(f"Inferring per-file on {project} @ {relative}")
                    reldf: pt.DataFrame[InferredSchema] = self._infer_file(
                        project, relative
                    )
                    updates.append(reldf)
        if updates:
            self.inferred = (
                pd.concat([self.inferred, *updates], ignore_index=True)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

    @abc.abstractmethod
    def _infer_file(
        self, project: pathlib.Path, relative: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        pass
