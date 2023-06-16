import concurrent.futures
import pathlib
import shutil

import click
import pandas
import tqdm

from scripts.common import output
from scripts.common.schemas import TypeCollectionCategory, TypeCollectionSchema
from scripts.infer.structure import DatasetFolderStructure


from scripts.utils import (
    format_parallel_exec_result,
    scratchpad,
    worker_count,
)

from .inference import Inference, factory, SUPPORTED_TOOLS

from libcst import codemod
from libcst._exceptions import ParserSyntaxError


@click.command(
    name="infer",
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
@click.option(
    "-e",
    "--task",
    type=click.Choice(
        [
            str(TypeCollectionCategory.VARIABLE),
            str(TypeCollectionCategory.CALLABLE_PARAMETER),
            str(TypeCollectionCategory.CALLABLE_RETURN),
        ],
    ),
    help="Remove and infer annotations in the codebase",
    required=True,
)
# @click.option("-a", "--annotate", is_flag=True, help="Add inferred annotations back into codebase")
def cli_entrypoint(
    tool: type[Inference],
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
    task: str,
) -> None:
    structure = DatasetFolderStructure(dataset)
    print("Dataset Kind:", structure)

    task = TypeCollectionCategory.__getitem__(task)

    inference_tool = tool()
    test_set = structure.test_set()

    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
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
                f"Skipping {project}, results are already at {inference_output}, and --overwrite was not given!"
            )
            continue

        with (
            scratchpad(project) as sc,
            inference_tool.activate_logging(sc),
        ):
            print(f"Preprocessing repo by removing {task} annotations and other tool-specificities")
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=inference_tool.preprocessor(task=task),
                jobs=worker_count(),
                files=[sc / s for s in subset],
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
                continue

            except ParserSyntaxError:
                inference_tool.logger.error("Failed to parse project", exc_info=True)
                continue

            except Exception:
                inference_tool.logger.error(f"Unhandled error occurred", exc_info=True)
                continue

            else:
                with pandas.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.expand_frame_repr",
                    False,
                ):
                    inferred = inferred[inferred[TypeCollectionSchema.category] == task]
                    print(inferred.sample(n=min(len(inferred), 20)).sort_index())

                inference_io.write(artifact=inferred)

            finally:
                # Copy generated log files
                for log_path in (
                    output.InferredLoggingIO.info_log_path,
                    output.InferredLoggingIO.debug_log_path,
                    output.InferredLoggingIO.error_log_path,
                ):
                    shutil.copy(log_path(sc), log_path(inference_output.parent))
                print(f"Logs have been stored at {inference_output.parent}")

            """ if annotate:
                # Copy original project
                shutil.copytree(
                    inpath,
                    outdir,
                    ignore_dangling_symlinks=True,
                    symlinks=True,
                    dirs_exist_ok=True,
                )

                # Reremove annotations
                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypeAnnotationRemover(
                        context=codemod.CodemodContext(),
                        variables=TypeCollectionCategory.VARIABLE in tasked,
                        parameters=TypeCollectionCategory.CALLABLE_PARAMETER in tasked,
                        rets=TypeCollectionCategory.CALLABLE_RETURN in tasked,
                    ),
                    jobs=worker_count(),
                    files=codemod.gather_files([str(outdir)]),
                    repo_root=str(outdir),
                )

                print(
                    format_parallel_exec_result(
                        action="Annotation Removal Preservation (in case inference mutated codebase)",
                        result=result,
                    )
                )

                print(f"Applying Annotations to codebase at {outdir}")
                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypeAnnotationApplierTransformer(
                        codemod.CodemodContext(), top_preds_only(inferred)
                    ),
                    files=codemod.gather_files([str(outdir)]),
                    jobs=worker_count(),
                    repo_root=str(outdir),
                )
                print(format_parallel_exec_result(action="Annotation Application", result=result)) """


if __name__ == "__main__":
    cli_entrypoint()
