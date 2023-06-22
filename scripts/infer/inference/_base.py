import abc
import contextlib
import logging
import pathlib
import sys
import typing
import weakref
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional

import pandas as pd
import pandera.typing as pt
from libcst import codemod

from scripts.common.schemas import InferredSchema, TypeCollectionCategory
from scripts.infer.inference import _utils
from scripts.infer.structure import DatasetFolderStructure


class Inference(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

        self.logger = logging.getLogger(type(self).__qualname__)
        self.logger.setLevel(logging.INFO)
        self.artifacts: list[typing.Any] | None = None

    @abc.abstractmethod
    def preprocessor(self, task: TypeCollectionCategory) -> codemod.Codemod:
        ...

    @abc.abstractmethod
    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> pt.DataFrame[InferredSchema]:
        pass

    @contextlib.contextmanager
    def activate_artifact_tracking(self, location: pathlib.Path, dataset: DatasetFolderStructure, repository: pathlib.Path, task: TypeCollectionCategory) -> None:
        from scripts.common.output import InferenceArtifactIO

        self.artifacts = list[typing.Any]()
        yield

        io = InferenceArtifactIO(
            artifact_root=location,
            dataset=dataset,
            repository=repository,
            tool_name=self.method(),
            task=task
        )
        self.logger.info(f"Writing inference artifacts to {io.full_location()}")
        io.write(self.artifacts)
        self.artifacts = None

    def register_artifact(self, artifact: typing.Any) -> None:
        if self.artifacts is None:
            self.logger.warning(f"Not registering artifact; contextmanager is not active")

        else:
            self.artifacts.append(artifact)

    @contextlib.contextmanager
    def activate_logging(self, project: pathlib.Path) -> typing.Generator[None, None, None]:
        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        from scripts.common.output import InferredLoggingIO

        generic_sout_handler = logging.StreamHandler(stream=sys.stdout)
        generic_sout_handler.setLevel(logging.INFO)
        generic_sout_handler.setFormatter(fmt=formatter)

        generic_filehandler = logging.FileHandler(
            filename=InferredLoggingIO.info_log_path(project), mode="w"
        )
        generic_filehandler.setLevel(logging.INFO)
        generic_filehandler.setFormatter(fmt=formatter)

        debug_handler = logging.FileHandler(
            filename=InferredLoggingIO.debug_log_path(project), mode="w"
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(filter=lambda record: record.levelno == logging.DEBUG)
        debug_handler.setFormatter(fmt=formatter)

        error_handler = logging.FileHandler(
            filename=InferredLoggingIO.error_log_path(project), mode="w"
        )
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


class ParallelisableInference(ProjectWideInference, ABC):
    def __init__(
        self,
        cpu_executor: ProcessPoolExecutor | None = None,
        model_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__()
        self.cpu_executor = cpu_executor or _utils.cpu_executor()
        self.model_executor = model_executor or _utils.model_executor()

        self.logger.info(
            f"Tool has {self.cpu_executor._max_workers} CPU subprocesses, {self.model_executor._max_workers} GPU subthreads"
        )

        self.shutdown_hook = weakref.finalize(self, self._shutdown_hook)

    def _shutdown_hook(self) -> None:
        self.cpu_executor.shutdown(wait=True, cancel_futures=False)
        self.model_executor.shutdown(wait=True, cancel_futures=False)
