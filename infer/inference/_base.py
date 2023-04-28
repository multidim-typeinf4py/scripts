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
    PROJECT = enum.auto()

    @staticmethod
    def from_folderpath(path: pathlib.Path) -> "DatasetFolderStructure":
        if path.name.lower() == "many-types-4-py-dataset":
            return DatasetFolderStructure.MANYTYPES4PY
        elif path.name.lower() == "typilus":
            return DatasetFolderStructure.TYPILUS
        else:
            return DatasetFolderStructure.PROJECT

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
        elif self == DatasetFolderStructure.PROJECT:
            yield dataset_root

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
        else:
            raise RuntimeError("Cannot determine author or repo of a simple folder")

    def test_set(
        self, dataset_root: pathlib.Path
    ) -> dict[pathlib.Path, set[pathlib.Path]]:
        if self == DatasetFolderStructure.MANYTYPES4PY:
            splits = pd.read_csv(
                dataset_root / "data" / "dataset_split.csv",
                header=None,
                names=["split", "filepath"],
            )
            test_split = splits[splits["split"] == "test"]["filepath"]

            test_set: dict[pathlib.Path, set[pathlib.Path]] = {}
            for key, g in (
                test_split.str.strip(to_strip='"')
                .str.removeprefix("repos/")
                .str.split(pat=r"\/", n=2, expand=True)
                .set_axis(["author", "project", "file"], axis=1)
                .groupby(by=["author", "project"])
            ):
                author, project = key
                test_set[dataset_root / author / project] = set(
                    map(pathlib.Path, g["file"].tolist())
                )
            return test_set

        elif self == DatasetFolderStructure.TYPILUS:

            def typilus_impl(
                path2typilus: pathlib.Path,
            ) -> dict[pathlib.Path, list[str]]:
                ...

            return typilus_impl(dataset_root)

        elif self == DatasetFolderStructure.PROJECT:
            return dict()


class Inference(abc.ABC):
    inferred: pt.DataFrame[InferredSchema]
    cache_storage: dict[pathlib.Path, typing.Any]

    def __init__(
        self,
        cache: Optional[pathlib.Path],
    ) -> None:
        super().__init__()
        self.cache = cache.resolve() if cache else None
        self.inferred = InferredSchema.example(size=0)

        self.logger = logging.getLogger(type(self).__qualname__)
        self.cache_storage: dict[pathlib.Path, typing.Any] = dict()

    @abc.abstractmethod
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> None:
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
        return self.cache / self.method

    @property
    @abc.abstractmethod
    def method(self) -> str:
        pass


class ProjectWideInference(Inference):
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> None:
        self.logger.debug(f"Inferring project-wide on {mutable}")
        self.inferred = self._infer_project(mutable)

        self._write_cache()

    @abc.abstractmethod
    def _infer_project(
        self, root: pathlib.Path, subset: Optional[set[pathlib.Path]] = None
    ) -> pt.DataFrame[InferredSchema]:
        pass


class PerFileInference(Inference):
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> None:
        updates = list()

        if subset is not None:
            paths = map(lambda p: mutable / p, subset)
        else:
            paths = mutable.rglob("*.py")

        for subfile in paths:
            relative = subfile.relative_to(mutable)
            self.logger.debug(f"Inferring per-file on {mutable} @ {relative}")
            reldf: pt.DataFrame[InferredSchema] = self._infer_file(mutable, relative)
            updates.append(reldf)

        if updates:
            self.inferred = pd.concat(
                [self.inferred, *updates], ignore_index=True
            ).pipe(pt.DataFrame[InferredSchema])

        self._write_cache()

    @abc.abstractmethod
    def _infer_file(
        self, root: pathlib.Path, relative: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        pass
