import concurrent.futures
import dataclasses
import json
import pathlib
import shutil

import click
import pandas
import tqdm

from scripts.common import output, extending
from scripts.common.schemas import (
    TypeCollectionCategory,
    TypeCollectionSchema,
    ExtendedInferredSchema,
)
from scripts.infer.structure import DatasetFolderStructure

from pandera import typing as pt

from scripts.utils import (
    format_parallel_exec_result,
    scratchpad,
    worker_count,
)

from scripts.infer.inference import Inference, factory, SUPPORTED_TOOLS
from .remover import TT5AllAnnotRemover

from libcst import codemod
from libcst._exceptions import ParserSyntaxError


@click.command(
    name="tt5exp",
    help="Apply an inference tool to the provided repository",
)
@click.option(
    "-t",
    "--tool",
    type=click.Choice(
        choices=list(SUPPORTED_TOOLS),
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: factory(value),
    required=True,
    help="Supported inference methods",
)
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Dataset to iterate over (can also be a singular project!)",
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Base folder for inference results to be written into",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output folder from previous run",
)
def cli_entrypoint(
    tool: type[Inference],
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
) -> None:
    import os

    structure = DatasetFolderStructure(dataset)

    os.environ["ARTIFACT_ROOT"] = str(outpath)
    os.environ["DATASET_ROOT"] = str(structure.dataset_root)
    task = "all"

    os.environ["TASK"] = task

    print("Dataset Kind:", structure)

    inference_tool = tool()
    test_set = structure.test_set()

    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        os.environ["REPOSITORY"] = str(project)
        pbar.set_description(desc=f"{project}")

        inference_io = output.InferredIO(
            artifact_root=outpath,
            dataset=structure,
            repository=project,
            tool_name=inference_tool.method(),
            task=task,
        )
        inference_output = inference_io.full_location()

        print(f"Selecting {inference_output} as the output folder")

        # Skip if we are not overwriting results
        if not overwrite and inference_output.exists():
            print(
                f"Skipping inference for {project}, results are already at {inference_output}, and --overwrite was not given!"
            )

        else:
            with (
                scratchpad(project) as sc,
                inference_tool.activate_logging(sc),
                inference_tool.activate_artifact_tracking(
                    location=outpath, dataset=structure, repository=project, task=task
                ),
            ):
                print(
                    f"Preprocessing repo by removing {task} annotations and other tool-specificities on ALL files"
                )
                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TT5AllAnnotRemover(context=codemod.CodemodContext()),
                    jobs=worker_count(),
                    files=codemod.gather_files([str(sc)]),
                    repo_root=str(sc),
                )
                print(format_parallel_exec_result(action="Preprocessing", result=result))

                # Run inference task for hour before aborting
                # print("Starting inference task with 1h timeout")
                try:
                    inferred = inference_tool.infer(sc, project, subset)

                except concurrent.futures.TimeoutError:
                    inference_tool.logger.error(
                        "Took over an hour to infer types, killing inference subprocess. "
                        "Results will NOT be written to disk",
                        exc_info=True,
                    )
                    break

                except ParserSyntaxError:
                    inference_tool.logger.error(f"{project} - Failed to parse", exc_info=True)

                    #dump_folder = pathlib.Path.cwd() / ".broken"
                    #inference_tool.logger.error(
                    #    f"Dumping {sc} @ {dump_folder} for further examination",
                    #    exc_info=True,
                    #)
                    #shutil.copytree(sc, dump_folder, dirs_exist_ok=True)

                    break

                except Exception:
                    inference_tool.logger.error(
                        f"{project} - Unhandled error occurred", exc_info=True
                    )

                    #dump_folder = pathlib.Path.cwd() / ".broken"
                    #inference_tool.logger.error(
                    #    f"Dumping {sc} @ {dump_folder} for further examination",
                    #    exc_info=True,
                    #)
                    #shutil.copytree(sc, dump_folder, dirs_exist_ok=True)

                    break

                else:
                    with pandas.option_context(
                        "display.max_rows",
                        None,
                        "display.max_columns",
                        None,
                        "display.expand_frame_repr",
                        False,
                    ):
                        print(inferred.sample(n=min(len(inferred), 20)).sort_index())

                    inference_io.write(artifact=inferred)

                finally:
                    # Copy generated log files
                    inference_output.parent.mkdir(parents=True, exist_ok=True)
                    for log_path in (
                        output.InferredLoggingIO.info_log_path,
                        output.InferredLoggingIO.debug_log_path,
                        output.InferredLoggingIO.error_log_path,
                    ):
                        shutil.copy(log_path(sc), log_path(inference_output.parent))
                    print(f"Logs have been stored at {inference_output.parent}")


if __name__ == "__main__":
    tqdm.pandas()
    cli_entrypoint()
