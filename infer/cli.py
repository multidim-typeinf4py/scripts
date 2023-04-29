import pathlib
import shutil
from typing import Optional

import click
import pandas
import tqdm

from common.annotations import TypeAnnotationRemover
from common import output
from common.schemas import TypeCollectionCategory, TypeCollectionSchema
from infer.inference._base import DatasetFolderStructure

from infer.insertion import TypeAnnotationApplierTransformer

from utils import format_parallel_exec_result, scratchpad, top_preds_only

from .inference import Inference, factory, SUPPORTED_TOOLS

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
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Dataset to iterate over (can also be a singular project!)",
)
@click.option(
    "-c",
    "--cache-path",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help="Folder to put ML inference cache inside of",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output folder from previous run",
)
@click.option(
    "-r",
    "--remove",
    type=click.Choice(
        [
            str(TypeCollectionCategory.VARIABLE),
            str(TypeCollectionCategory.CALLABLE_PARAMETER),
            str(TypeCollectionCategory.CALLABLE_RETURN),
        ],
    ),
    help="Remove annotations in the codebase",
    multiple=True,
    required=True,
)
@click.option(
    "-i",
    "--infer",
    type=click.Choice(
        [
            str(TypeCollectionCategory.VARIABLE),
            str(TypeCollectionCategory.CALLABLE_PARAMETER),
            str(TypeCollectionCategory.CALLABLE_RETURN),
        ],
    ),
    help="Infer annotations in the codebase",
    multiple=True,
    required=True,
)
@click.option("-a", "--annotate", is_flag=True, help="Add inferred annotations back into codebase")
def cli_entrypoint(
    tool: type[Inference],
    dataset: pathlib.Path,
    cache_path: Optional[pathlib.Path],
    overwrite: bool,
    remove: list[str],
    infer: list[str],
    annotate: bool,
) -> None:
    removing = list(map(TypeCollectionCategory.__getitem__, remove))
    inferring = list(map(TypeCollectionCategory.__getitem__, infer))

    if illegal := set(inferring) - set(removing):
        print(
            f"Refusing to perform inference; asking to infer {illegal}, while not removing them will deliver "
            f"inaccurate results"
        )
        return

    structure = DatasetFolderStructure.from_folderpath(dataset)
    print(dataset, structure)
    test_set = structure.test_set(dataset)

    projects = list(structure.project_iter(dataset))
    for inpath in (pbar := tqdm.tqdm(projects)):
        if not (subset := test_set.get(inpath, set())):
            continue
        pbar.set_description(desc=f"Inferring over {inpath}")

        with scratchpad(inpath) as sc:
            print(f"Using {sc} as a scratchpad for inference!")

            if removing:
                print(f"annotation removal flag provided, removing annotations on '{sc}'")
                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypeAnnotationRemover(
                        context=codemod.CodemodContext(),
                        variables=TypeCollectionCategory.VARIABLE in removing,
                        parameters=TypeCollectionCategory.CALLABLE_PARAMETER in removing,
                        rets=TypeCollectionCategory.CALLABLE_RETURN in removing,
                    ),
                    files=codemod.gather_files([str(sc)]),
                    repo_root=str(sc),
                )
                print(format_parallel_exec_result(action="Annotation Removal", result=result))

            inference_tool = tool(cache=cache_path)
            inference_tool.infer(mutable=sc, readonly=inpath, subset=subset)

            outdir = output.inference_output_path(
                dataset,
                tool=inference_tool.method,
                removed=removing,
                inferred=inferring,
            )
            if outdir.is_dir() and overwrite:
                shutil.rmtree(outdir)

            elif outdir.is_dir() and not overwrite:
                raise RuntimeError(
                    f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
                )

            print(f"Inference completed; writing results to {outdir}")

        # Copy original project and re-remove annotations
        shutil.copytree(inpath, outdir, symlinks=True)
        if removing:
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=TypeAnnotationRemover(
                    context=codemod.CodemodContext(),
                    variables=TypeCollectionCategory.VARIABLE in removing,
                    parameters=TypeCollectionCategory.CALLABLE_PARAMETER in removing,
                    rets=TypeCollectionCategory.CALLABLE_RETURN in removing,
                ),
                files=codemod.gather_files([str(outdir)]),
                repo_root=str(outdir),
            )
            print(
                format_parallel_exec_result(
                    action="Annotation Removal Preservation (in case inference mutated codebase)",
                    result=result,
                )
            )

        with pandas.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.expand_frame_repr",
            False,
        ):
            df = inference_tool.inferred.copy(deep=True)
            df = df[df[TypeCollectionSchema.category].isin(inferring)]

            print(df.sample(n=min(len(df), 20)).sort_index())

        output.write_inferred(df, outdir)
        print(f"Inferred types have been stored at {outdir}")

        if annotate:
            print(f"Applying Annotations to codebase at {outdir}")
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=TypeAnnotationApplierTransformer(
                    codemod.CodemodContext(), top_preds_only(df)
                ),
                files=codemod.gather_files([str(outdir)]),
                # jobs=1,
                repo_root=str(outdir),
            )
            print(format_parallel_exec_result(action="Annotation Application", result=result))


if __name__ == "__main__":
    cli_entrypoint()
