import pathlib
import shutil

import click
from common import output
from common import factory

from infer.insertion import TypeAnnotationApplierTransformer
from infer.removal import HintRemover

from utils import format_parallel_exec_result, scratchpad, top_preds_only

from .inference import Inference, MyPy, PyreInfer, PyreQuery, TypeWriter, Type4Py, HiTyper

from libcst import codemod


@click.command(
    name="infer",
    help="Apply an inference tool to the provided repository",
)
@click.option(
    "-t",
    "--tool",
    type=click.Choice(
        choices=[
            MyPy.__name__.lower(),
            PyreInfer.__name__.lower(),
            PyreQuery.__name__.lower(),
            HiTyper.__name__.lower(),
            TypeWriter.__name__.lower(),
            Type4Py.__name__.lower(),
        ],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: factory._inference_factory(value),
    required=True,
    help="Supported inference methods",
)
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Project to infer over",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output folder. Has no effect if --output was not given",
)
@click.option(
    "-r",
    "--remove-annos",
    is_flag=True,
    help="Remove all annotations in the codebase before inferring",
)
def cli_entrypoint(
    tool: type[Inference],
    inpath: pathlib.Path,
    overwrite: bool,
    remove_annos: bool,
) -> None:
    with scratchpad(inpath) as sc:
        print(f"Using {sc} as a scratchpad for inference!")

        if remove_annos:
            print(f"--remove-annos provided, removing annotations on '{sc}'")
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=HintRemover(codemod.CodemodContext()),
                files=codemod.gather_files([str(sc)]),
                repo_root=str(sc),
            )
            print(format_parallel_exec_result(action="Annotation Removal", result=result))

        inference_tool = tool(sc)
        inference_tool.infer()

        outdir = output.inference_output_path(inpath, tool=inference_tool.method)
        if outdir.is_dir() and overwrite:
            shutil.rmtree(outdir)

        elif outdir.is_dir() and not overwrite:
            raise RuntimeError(
                f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
            )

        print(f"Inference completed; writing results to {outdir}")
        shutil.copytree(sc, outdir, symlinks=True)

    output.write_icr(inference_tool.inferred, outdir)
    print(f"Inferred types have been stored at {outdir}")

    print(f"Applying Annotations to codebase at {outdir}")
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=TypeAnnotationApplierTransformer(
            codemod.CodemodContext(), top_preds_only(inference_tool.inferred)
        ),
        files=codemod.gather_files([str(outdir)]),
        jobs=1,
        repo_root=str(outdir),
    )
    print(format_parallel_exec_result(action="Annotation Application", result=result))


if __name__ == "__main__":
    cli_entrypoint()
