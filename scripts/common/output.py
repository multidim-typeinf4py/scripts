import abc
import json
import pickle
import typing

import pathlib

import pandas as pd
import pandera.typing as pt
from libcst import codemod

from scripts.common.schemas import (
    ContextCategory,
    ContextSymbolSchema,
    TypeCollectionCategory,
    TypeCollectionSchema,
    InferredSchema,
    ExtendedTypeCollectionSchema,
    ExtendedInferredSchema,
)
from scripts.infer.structure import DatasetFolderStructure


A = typing.TypeVar("A")


class ArtifactIO(abc.ABC, typing.Generic[A]):
    def __init__(self, artifact_root: pathlib.Path) -> None:
        super().__init__()
        self.artifact_root = artifact_root

    def read(self) -> A:
        return self._read(self.full_location())

    @abc.abstractmethod
    def _read(self, input_location: pathlib.Path) -> A:
        ...

    def write(self, artifact: A) -> None:
        outpath_path = self.full_location()
        outpath_path.parent.mkdir(parents=True, exist_ok=True)
        self._write(artifact, outpath_path)

    @abc.abstractmethod
    def _write(self, artifact: A, output_location: pathlib.Path) -> None:
        ...

    def full_location(self) -> pathlib.Path:
        return self.artifact_root / self.relative_location()

    @abc.abstractmethod
    def relative_location(self) -> pathlib.Path:
        ...


class DatasetDependentIO(ArtifactIO[A]):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure | str,
        repository: pathlib.Path,
    ) -> None:
        super().__init__(artifact_root)
        self.dataset = dataset
        self.repository = repository

    def relative_location(self) -> pathlib.Path:
        ar = self.dataset.author_repo(self.repository)
        dataset_name = (
            self.dataset
            if isinstance(self.dataset, str)
            else type(self.dataset).__name__
        )
        return pathlib.Path(dataset_name) / f"{ar!s}"


class ContextIO(DatasetDependentIO[pt.DataFrame[ContextSymbolSchema]]):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure | str,
        repository: pathlib.Path,
    ):
        super().__init__(artifact_root, dataset, repository)

    def _read(self, input_location: pathlib.Path) -> pt.DataFrame[ContextSymbolSchema]:
        return pd.read_csv(
            input_location,
            converters={
                ContextSymbolSchema.category: lambda c: TypeCollectionCategory[c],
                # ContextSymbolSchema.context_category: lambda c: ContextCategory(int(c)),
            },
            keep_default_na=False,
            na_values=[""],
        ).pipe(pt.DataFrame[ContextSymbolSchema])

    def _write(
        self, artifact: pt.DataFrame[ContextSymbolSchema], output_location: pathlib.Path
    ) -> None:
        return artifact.to_csv(output_location, index=False, na_rep="")

    def relative_location(self) -> pathlib.Path:
        return super().relative_location() / f"context.csv"


class InferredIO(DatasetDependentIO[pt.DataFrame[InferredSchema]]):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure,
        repository: pathlib.Path,
        tool_name: str,
        task: TypeCollectionCategory | str,
    ) -> None:
        super().__init__(artifact_root, dataset, repository)
        self.tool_name = tool_name
        self.task = str(task)

    def _read(self, input_location: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        return pd.read_csv(
            input_location,
            converters={
                TypeCollectionSchema.category: lambda c: TypeCollectionCategory[c]
            },
            keep_default_na=False,
            na_values=[""],
        ).pipe(pt.DataFrame[InferredSchema])

    def _write(
        self, artifact: pt.DataFrame[InferredSchema], output_location: pathlib.Path
    ) -> None:
        return artifact.to_csv(output_location, index=False, na_rep="")

    def relative_location(self) -> pathlib.Path:
        return (
            super().relative_location()
            / f"{self.tool_name}"
            / f"{str(self.task)}"
            / "inferred.csv"
        )


class DatasetIO(DatasetDependentIO[pt.DataFrame[TypeCollectionSchema]]):
    def _read(self, input_location: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        return pd.read_csv(
            input_location,
            converters={
                TypeCollectionSchema.category: lambda c: TypeCollectionCategory[c]
            },
            keep_default_na=False,
            na_values=[""],
        ).pipe(pt.DataFrame[TypeCollectionSchema])

    def _write(
        self,
        artifact: pt.DataFrame[TypeCollectionSchema],
        output_location: pathlib.Path,
    ) -> None:
        return artifact.to_csv(output_location, index=False, na_rep="")

    def relative_location(self) -> pathlib.Path:
        return super().relative_location() / "ground_truth.csv"


class InferredLoggingIO:
    @staticmethod
    def error_log_path(outpath: pathlib.Path) -> pathlib.Path:
        return outpath / "log.err"

    @staticmethod
    def debug_log_path(outpath: pathlib.Path) -> pathlib.Path:
        return outpath / "log.dbg"

    @staticmethod
    def info_log_path(outpath: pathlib.Path) -> pathlib.Path:
        return outpath / "log.inf"


class ExtendedDatasetIO(DatasetDependentIO[pt.DataFrame[ExtendedTypeCollectionSchema]]):
    def _read(
        self, input_location: pathlib.Path
    ) -> pt.DataFrame[ExtendedTypeCollectionSchema]:
        return pd.read_csv(
            input_location,
            converters={
                TypeCollectionSchema.category: lambda c: TypeCollectionCategory[c]
            },
            keep_default_na=False,
            na_values=[""],
        ).pipe(pt.DataFrame[ExtendedTypeCollectionSchema])

    def _write(
        self,
        artifact: pt.DataFrame[ExtendedTypeCollectionSchema],
        output_location: pathlib.Path,
    ) -> None:
        return artifact.to_csv(output_location, index=False, na_rep="")

    def relative_location(self) -> pathlib.Path:
        return super().relative_location() / "extended_ground_truth.csv"


class ExtendedInferredIO(DatasetDependentIO[pt.DataFrame[ExtendedInferredSchema]]):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure,
        repository: pathlib.Path,
        tool_name: str,
        task: TypeCollectionCategory,
    ) -> None:
        super().__init__(artifact_root, dataset, repository)
        self.tool_name = tool_name
        self.task = task

    def _read(self, input_location: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        return pd.read_csv(
            input_location,
            converters={
                TypeCollectionSchema.category: lambda c: TypeCollectionCategory[c]
            },
            keep_default_na=False,
            na_values=[""],
        ).pipe(pt.DataFrame[InferredSchema])

    def _write(
        self, artifact: pt.DataFrame[InferredSchema], output_location: pathlib.Path
    ) -> None:
        return artifact.to_csv(output_location, index=False, na_rep="")

    def relative_location(self) -> pathlib.Path:
        return (
            super().relative_location()
            / f"{self.tool_name}"
            / f"{str(self.task)}"
            / "extended_inferred.csv"
        )


class InferenceArtifactIO(DatasetDependentIO[list[typing.Any]]):
    def __init__(
        self,
        artifact_root: pathlib.Path,
        dataset: DatasetFolderStructure | str,
        repository: pathlib.Path,
        tool_name: str,
        task: TypeCollectionCategory | str,
    ) -> None:
        super().__init__(artifact_root, dataset, repository)
        self.tool_name = tool_name
        self.task = str(task)

    def _read(self, input_location: pathlib.Path) -> list[typing.Any]:
        with input_location.open("rb") as f:
            return pickle.load(f)

    def _write(self, artifact: list[typing.Any], output_location: pathlib.Path) -> None:
        with output_location.open("wb") as f:
            pickle.dump(artifact, f)

    def relative_location(self) -> pathlib.Path:
        return (
            super().relative_location()
            / f"{self.tool_name}"
            / f"{str(self.task)}"
            / f"{self.tool_name}-artifacts.pickle"
        )
