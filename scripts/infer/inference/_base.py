import abc
import concurrent.futures
import contextlib
import enum
import pathlib
import sys
import typing
from typing import Optional

from scripts import utils
from scripts.common import output
from scripts.common.schemas import InferredSchema

import logging

from libcst import codemod

import pandera.typing as pt
import pandas as pd


class DatasetFolderStructure(enum.Enum):
    MANYTYPES4PY = enum.auto()
    BETTERTYPES4PY = enum.auto()
    PROJECT = enum.auto()

    @staticmethod
    def from_folderpath(path: pathlib.Path) -> "DatasetFolderStructure":
        if path.name.lower() == "many-types-4-py-dataset":
            return DatasetFolderStructure.MANYTYPES4PY
        elif path.name.lower() == "better-types-4-py-dataset":
            return DatasetFolderStructure.BETTERTYPES4PY
        else:
            return DatasetFolderStructure.PROJECT

    def project_iter(
        self, dataset_root: pathlib.Path
    ) -> typing.Generator[pathlib.Path, None, None]:
        if self == DatasetFolderStructure.MANYTYPES4PY:
            repo_suffix = dataset_root / "repos"
            authors = (
                author
                for author in repo_suffix.iterdir()
                if author.is_dir() and not author.name.startswith(".")
            )
            for author in authors:
                repos = (repo for repo in author.iterdir() if repo.is_dir())
                yield from repos

        elif self == DatasetFolderStructure.BETTERTYPES4PY:
            repo_suffix = dataset_root / "repos" / "test"
            repos = (repo for repo in repo_suffix.iterdir() if repo.is_dir())
            yield from repos
        elif self == DatasetFolderStructure.PROJECT:
            yield dataset_root

    def author_repo(self, repo: pathlib.Path) -> dict:
        if self == DatasetFolderStructure.MANYTYPES4PY:
            return {"author": repo.parent.name, "repo": repo.name}
        elif self == DatasetFolderStructure.BETTERTYPES4PY:
            return dict(
                zip(
                    ("author", "repo"),
                    repo.name.split("__"),
                    strict=True,
                )
            )
        else:
            return {"author": repo.name, "repo": repo.name}

    def test_set(self, dataset_root: pathlib.Path) -> dict[pathlib.Path, set[pathlib.Path]]:
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
                test_set[dataset_root / "repos" / author / project] = set(
                    map(pathlib.Path, g["file"].tolist())
                )
            return test_set

        elif self == DatasetFolderStructure.BETTERTYPES4PY:
            repo_suffix = dataset_root / "repos" / "test"

            mapping = dict[pathlib.Path, set[pathlib.Path]]()

            for repo in repo_suffix.iterdir():
                if repo.is_dir() and (fs := codemod.gather_files([repo_suffix / repo])):
                    mapping[repo_suffix / repo] = set(
                        map(
                            lambda p: pathlib.Path(p).relative_to(repo_suffix / repo),
                            fs,
                        )
                    )

            return mapping

        elif self == DatasetFolderStructure.PROJECT:
            subfiles = set(
                map(
                    lambda p: pathlib.Path(p).relative_to(dataset_root),
                    codemod.gather_files([str(dataset_root)]),
                )
            )
            return {dataset_root: subfiles}


class Inference(abc.ABC):
    def __init__(
        self,
        cpu_executor: concurrent.futures.ProcessPoolExecutor | None = None,
        model_executor: concurrent.futures.ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger(type(self).__qualname__)
        self.logger.setLevel(logging.INFO)

        self._cpu_executor = cpu_executor
        self._model_executor = model_executor

    @contextlib.contextmanager
    def cpu_executor(self) -> typing.Generator[concurrent.futures.ProcessPoolExecutor, None, None]:
        if self._cpu_executor is None:
            with concurrent.futures.ProcessPoolExecutor(max_workers=utils.worker_count()) as pool:
                yield pool

        else:
            yield self._cpu_executor

    @contextlib.contextmanager
    def model_executor(self) -> typing.Generator[concurrent.futures.ThreadPoolExecutor, None, None]:
        if self._model_executor is None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                yield pool

        else:
            yield self._model_executor

    @abc.abstractmethod
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        pass

    @contextlib.contextmanager
    def activate_logging(self, project: pathlib.Path) -> typing.Generator[None, None, None]:
        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        generic_sout_handler = logging.StreamHandler(stream=sys.stdout)
        generic_sout_handler.setLevel(logging.INFO)
        generic_sout_handler.setFormatter(fmt=formatter)

        generic_filehandler = logging.FileHandler(filename=output.info_log_path(project), mode="w")
        generic_filehandler.setLevel(logging.INFO)
        generic_filehandler.setFormatter(fmt=formatter)

        debug_handler = logging.FileHandler(filename=output.debug_log_path(project), mode="w")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(filter=lambda record: record.levelno == logging.DEBUG)
        debug_handler.setFormatter(fmt=formatter)

        error_handler = logging.FileHandler(filename=output.error_log_path(project), mode="w")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(fmt=formatter)

        for handler in (generic_filehandler, debug_handler, error_handler):
            self.logger.addHandler(hdlr=handler)

        yield

        for handler in (generic_filehandler, debug_handler, error_handler):
            self.logger.removeHandler(hdlr=handler)

    @abc.abstractmethod
    def method(self) -> str:
        pass


class ProjectWideInference(Inference):
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        self.logger.info(f"Inferring project-wide for {readonly}")

        if subset is not None:
            subset = {s for s in subset if (mutable / s).is_file()}
        else:
            subset = set(
                map(
                    lambda r: pathlib.Path(r).relative_to(mutable),
                    codemod.gather_files([str(mutable)]),
                )
            )

        inferred = self._infer_project(mutable, subset)
        self.logger.info("Inference completed")
        return inferred

    @abc.abstractmethod
    def _infer_project(
        self, mutable: pathlib.Path, subset: set[pathlib.Path]
    ) -> pt.DataFrame[InferredSchema]:
        pass


class PerFileInference(Inference):
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        updates = [InferredSchema.example(size=0)]

        if subset is not None:
            paths = map(lambda p: mutable / p, subset)
        else:
            paths = mutable.rglob("*.py")

        for subfile in paths:
            relative = subfile.relative_to(mutable)
            self.logger.info(f"Inferring per-file on {relative} @ {mutable} ({readonly})")
            reldf: pt.DataFrame[InferredSchema] = self._infer_file(mutable, relative)
            updates.append(reldf)

        self.logger.info("Inference completed")
        return pd.concat(updates, ignore_index=True).pipe(pt.DataFrame[InferredSchema])

    @abc.abstractmethod
    def _infer_file(
        self, root: pathlib.Path, relative: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        pass
