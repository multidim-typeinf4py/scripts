import abc
import enum
import pathlib
import pickle
import typing
from typing import Optional

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

    def author_repo(self, repo: pathlib.Path) -> dict:
        if self == DatasetFolderStructure.MANYTYPES4PY:
            return {"author": repo.parent.name, "repo": repo.name}
        elif self == DatasetFolderStructure.TYPILUS:
            return dict(
                zip(
                    ("author", "repo"),
                    repo.name.split("."),
                    strict=True,
                )
            )


class Inference(abc.ABC):
    inferred: pt.DataFrame[InferredSchema]
    cache_storage: dict[pathlib.Path, typing.Any]

    def __init__(self, mutable: pathlib.Path, readonly: pathlib.Path, cache: Optional[pathlib.Path]) -> None:
        super().__init__()
        self.mutable = mutable.resolve()
        self.readonly = readonly.resolve()
        self.cache = cache.resolve() if cache else None
        self.inferred = InferredSchema.example(size=0)

        self.logger = logging.getLogger(type(self).__qualname__)
        self.cache_storage: dict[pathlib.Path, typing.Any] = dict()

    @abc.abstractmethod
    def infer(self) -> None:
        pass

    def register_cache(self, relative: pathlib.Path, data: typing.Any) -> None:
        if not self.cache:
            print("WARNING: Cache argument was not supplied, skipping registration...")
        else:
            assert relative not in self.cache_storage
            self.cache_storage[relative] = data

    def _write_cache(self) -> None:
        if not self.cache:
            print("WARNING: Cache argument was not supplied, skipping writing...")
        else:
            outpath = self._cache_path()
            outpath.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing {self.method}'s cache to {outpath}")

            with outpath.open("wb") as f:
                pickle.dump(self.cache_storage, f)

    def _load_cache(self) -> dict[pathlib.Path, typing.Any]:
        if not self.cache:
            print("WARNING: Cache argument was not supplied, assuming empty collection")
            return dict()

        inpath = self._cache_path()
        print(f"Loading cache from {inpath}")

        try:
            with inpath.open("rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Cache not found, assuming empty collection")
            return dict()

    def _cache_path(self) -> pathlib.Path:
        return self.cache / f"{self.readonly.name}@{self.method}"

    @property
    @abc.abstractmethod
    def method(self) -> str:
        pass


class ProjectWideInference(Inference):
    def infer(self) -> None:
        self.logger.debug(f"Inferring project-wide on {self.mutable}")
        self.inferred = self._infer_project()

        self._write_cache()

    @abc.abstractmethod
    def _infer_project(self) -> pt.DataFrame[InferredSchema]:
        pass


class PerFileInference(Inference):
    def infer(self) -> None:
        updates = list()
        for subfile in self.mutable.rglob("*.py"):
            relative = subfile.relative_to(self.mutable)
            if str(relative) not in self.inferred["file"]:
                self.logger.debug(f"Inferring per-file on {self.mutable} @ {relative}")
                reldf: pt.DataFrame[InferredSchema] = self._infer_file(relative)
                updates.append(reldf)
        if updates:
            self.inferred = (
                pd.concat([self.inferred, *updates], ignore_index=True)
                .reindex(columns=InferredSchemaColumns)
                .pipe(pt.DataFrame[InferredSchema])
            )

        self._write_cache()

    @abc.abstractmethod
    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        pass
