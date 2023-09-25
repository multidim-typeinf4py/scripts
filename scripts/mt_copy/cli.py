import os
import pathlib, shutil

import click
from libcst import codemod

from scripts.infer.preprocessers import monkey
from scripts import utils


@click.command(
    name="mt-unannotate",
    help="Preprocess for FlaPy",
)
@click.option(
    "-p",
    "--path",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
)
def cli_entrypoint(path: pathlib.Path) -> None:
    files = codemod.gather_files([str(path)])
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=monkey.MonkeyPreprocessor(codemod.CodemodContext()),
        files=list(filter(os.path.isfile, files)),
        jobs=utils.worker_count(),
        repo_root=str(path),
    )
    print(utils.format_parallel_exec_result(action="MT-Unannotate", result=result))
