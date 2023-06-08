import concurrent.futures
import pathlib
import shutil

import click
import pandas
import tqdm

from scripts.common.annotations import TypeAnnotationRemover
from scripts.common import output
from scripts.common.schemas import TypeCollectionCategory, TypeCollectionSchema
from scripts.infer.inference._base import DatasetFolderStructure

from scripts.infer.insertion import TypeAnnotationApplierTransformer

from scripts.utils import (
    format_parallel_exec_result,
    scratchpad,
    top_preds_only,
    worker_count,
)

from .inference import Inference, factory, SUPPORTED_TOOLS

import torch.multiprocessing as mp
from libcst import codemod


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
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Dataset to iterate over (can also be a singular project!)",
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
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
    multiple=True,
    required=True,
)
# @click.option("-a", "--annotate", is_flag=True, help="Add inferred annotations back into codebase")
def cli_entrypoint(
    tool: type[Inference],
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
    task: list[str],
) -> None:
    annotate = False
    tasked = list(map(TypeCollectionCategory.__getitem__, task))

    structure = DatasetFolderStructure.from_folderpath(dataset)
    print(dataset, structure)

    mp.set_start_method("spawn")

    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count() + 1
        ) as cpu_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as model_executor,
    ):
        inference_tool = tool(cpu_executor, model_executor)
        test_set = {p: s for p, s in structure.test_set(dataset).items() if p.is_dir()}

        for project, subset in (pbar := tqdm.tqdm(test_set.items())):
            pbar.set_description(desc=f"Inferring over {project}")

            ar = structure.author_repo(project)
            author_repo = f"{ar['author']}.{ar['repo']}"
            outdir = output.inference_output_path(
                outpath / author_repo,
                tool=inference_tool.method(),
                removed=tasked,
            )

            # Skip if we are not overwriting results
            if outdir.is_dir() and not overwrite:
                print(
                    f"Skipping {project}, results are already at {outdir}, and --overwrite was not given!"
                )
                continue

            inpath = project
            with (
                scratchpad(inpath) as sc,
                inference_tool.activate_logging(sc),
            ):
                print(f"Using {sc} as a scratchpad for inference!")
                if tasked:
                    print(
                        f"annotation removal flag provided, removing annotations on '{sc}'"
                    )
                    result = codemod.parallel_exec_transform_with_prettyprint(
                        transform=TypeAnnotationRemover(
                            context=codemod.CodemodContext(),
                            variables=TypeCollectionCategory.VARIABLE in tasked,
                            parameters=TypeCollectionCategory.CALLABLE_PARAMETER
                            in tasked,
                            rets=TypeCollectionCategory.CALLABLE_RETURN in tasked,
                        ),
                        jobs=worker_count(),
                        files=[sc / s for s in subset],
                        repo_root=str(sc),
                    )
                    print(
                        format_parallel_exec_result(
                            action="Annotation Removal", result=result
                        )
                    )

                # Run inference task for hour before aborting
                print("Starting inference task with 1h timeout")
                tasks = [
                    cpu_executor.submit(
                        type(inference_tool).infer,
                        inference_tool,
                        sc,
                        inpath,
                        subset,
                    )
                ]

                try:
                    for task in concurrent.futures.as_completed(tasks, timeout=60**2):
                        inferred = task.result()

                except concurrent.futures.TimeoutError as e:
                    inference_tool.logger.error(
                        "Took over an hour to infer types, killing inference subprocess. "
                        "Results will NOT be written to disk",
                        exc_info=True,
                    )
                    continue

                except Exception as e:
                    inference_tool.logger.error(
                        f"Unhandled error occurred", exc_info=True
                    )
                    continue

                else:
                    if outdir.is_dir() and overwrite:
                        shutil.rmtree(outdir)

                    outdir.mkdir(parents=True, exist_ok=True)
                    with pandas.option_context(
                        "display.max_rows",
                        None,
                        "display.max_columns",
                        None,
                        "display.expand_frame_repr",
                        False,
                    ):
                        inferred = inferred[
                            inferred[TypeCollectionSchema.category].isin(tasked)
                        ]
                        print(inferred.sample(n=min(len(inferred), 20)).sort_index())

                    output.write_inferred(inferred, outdir)
                    print(f"Inferred types have been stored at {outdir}")

                finally:
                    # Copy generated log files
                    outdir.mkdir(parents=True, exist_ok=True)
                    for log_path in (
                        output.info_log_path,
                        output.debug_log_path,
                        output.error_log_path,
                    ):
                        shutil.copy(log_path(sc), log_path(outdir))
                    print(f"Logs have been stored at {outdir}")

                if annotate:
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
                            parameters=TypeCollectionCategory.CALLABLE_PARAMETER
                            in tasked,
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
                    print(
                        format_parallel_exec_result(
                            action="Annotation Application", result=result
                        )
                    )


if __name__ == "__main__":
    cli_entrypoint()
