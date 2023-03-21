import pathlib
import shutil

import click
import pandas
from common.annotations import TypeAnnotationRemover
from common import output
from common import factory
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from infer.insertion import TypeAnnotationApplierTransformer

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
    help="Overwrite an existing output folder from previous run",
)
@click.option(
    "-rv",
    "--remove-var-annos",
    is_flag=True,
    help="Remove all variable annotations in the codebase before inferring",
)
@click.option(
    "-rp",
    "--remove-param-annos",
    is_flag=True,
    help="Remove all parameter annotations in the codebase before inferring",
)
@click.option(
    "-rr",
    "--remove-ret-annos",
    is_flag=True,
    help="Remove all return annotations in the codebase before inferring",
)
@click.option("-a", "--annotate", is_flag=True, help="Add inferred annotations back into codebase")
def cli_entrypoint(
    tool: type[Inference],
    inpath: pathlib.Path,
    overwrite: bool,
    remove_var_annos: bool,
    remove_param_annos: bool,
    remove_ret_annos: bool,
    annotate: bool,
) -> None:
    removing = []

    if remove_var_annos:
        removing.append(TypeCollectionCategory.VARIABLE)
        removing.append(TypeCollectionCategory.INSTANCE_ATTR)

    if remove_param_annos:
        removing.append(TypeCollectionCategory.CALLABLE_PARAMETER)

    if remove_ret_annos:
        removing.append(TypeCollectionCategory.CALLABLE_RETURN)

    with scratchpad(inpath) as sc:
        print(f"Using {sc} as a scratchpad for inference!")

        if remove_var_annos or remove_param_annos or remove_ret_annos:
            print(f"annotation removal flag provided, removing annotations on '{sc}'")
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=TypeAnnotationRemover(
                    context=codemod.CodemodContext(),
                    variables=remove_var_annos,
                    parameters=remove_param_annos,
                    rets=remove_ret_annos,
                ),
                files=codemod.gather_files([str(sc)]),
                repo_root=str(sc),
            )
            print(format_parallel_exec_result(action="Annotation Removal", result=result))

        inference_tool = tool(sc)
        inference_tool.infer()

        outdir = output.inference_output_path(inpath, tool=inference_tool.method, removed=removing)
        if outdir.is_dir() and overwrite:
            shutil.rmtree(outdir)

        elif outdir.is_dir() and not overwrite:
            raise RuntimeError(
                f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
            )

    print(f"Inference completed; writing results to {outdir}")

    shutil.copytree(inpath, outdir, symlinks=True)
    if remove_var_annos or remove_param_annos or remove_ret_annos:
        result = codemod.parallel_exec_transform_with_prettyprint(
            transform=TypeAnnotationRemover(
                context=codemod.CodemodContext(),
                variables=remove_var_annos,
                parameters=remove_param_annos,
                rets=remove_ret_annos,
            ),
            files=codemod.gather_files([str(inpath)]),
            repo_root=str(inpath),
        )
        print(
            format_parallel_exec_result(
                action="Annotation Removal Preservation (in case inference mutated codebase)",
                result=result,
            )
        )

    with pandas.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.expand_frame_repr", False
    ):
        df = inference_tool.inferred.copy(deep=True)
        df = df[
            df[TypeCollectionSchema.category].isin(removing) & df[TypeCollectionSchema.anno].notna()
        ]

        print(df.sample(n=min(len(df), 20)).sort_index())

    output.write_icr(df, outdir)
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
