from contextlib import contextmanager

import os
import pathlib
import tempfile
import typing
import shutil

from libcst.codemod import ParallelTransformResult

from common.schemas import InferredSchema
import pandera.typing as pt


@contextmanager
def scratchpad(untouched: pathlib.Path) -> typing.Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(
            src=str(untouched),
            dst=td,
            dirs_exist_ok=True,
            ignore_dangling_symlinks=True,
            symlinks=True,
        )
        try:
            yield pathlib.Path(td)
        finally:
            pass


@contextmanager
def working_dir(wd: pathlib.Path) -> typing.Generator[None, None, None]:
    oldcwd = pathlib.Path.cwd()
    os.chdir(str(wd))

    try:
        yield
    finally:
        os.chdir(str(oldcwd))


def format_parallel_exec_result(action: str, result: ParallelTransformResult) -> str:
    format = f"""=== {action} summary ===
    Finished codemodding {result.successes + result.skips + result.failures} files!
    
    Successful:     {result.successes}
    Failed:         {result.failures}
    Warnings:       {result.warnings}
    """

    return format


def top_preds_only(df: pt.DataFrame[InferredSchema]) -> pt.DataFrame[InferredSchema]:
    return df.loc[
        df.groupby(
            by=[InferredSchema.file, InferredSchema.category, InferredSchema.qname_ssa],
            sort=False,
        )[InferredSchema.topn].idxmin()
    ]


def worker_count() -> typing.Optional[int]:
    if cpt := os.getenv("SLURM_CPUS_PER_TASK"):
        return int(cpt)
    if cpt := os.getenv("MDTI4PY_CPUS"):
        return int(cpt)
    return os.cpu_count()
