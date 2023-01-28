from contextlib import contextmanager

import os
import pathlib
import tempfile
import typing
import shutil

from libcst.codemod._cli import ParallelTransformResult


@contextmanager
def scratchpad(untouched: pathlib.Path) -> typing.Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(
            src=str(untouched),
            dst=td,
            dirs_exist_ok=True,
            ignore_dangling_symlinks=True,
            symlinks=False,
        )
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


def format_parallel_exec_result(action: str, result: ParallelTransformResult) -> str:
    format = f"""=== {action} summary ===
    Finished codemodding {result.successes + result.skips + result.failures} files!
    
    Successful:     {result.successes}
    Failed:         {result.failures}
    Warnings:       {result.warnings}
    """

    return format