import enum
import pathlib

import click
from libcst import codemod

from src.common.schemas import InferredSchema, TypeCollectionCategory
from src.common import output

from src.context.features import RelevantFeatures
from src.infer.inference import Inference, factory, SUPPORTED_TOOLS
from src.infer.insertion import TypeAnnotationApplierTransformer
from utils import format_parallel_exec_result, scratchpad, worker_count

from .visitors import generate_context_vectors_for_project


class Purpose(str, enum.Enum):
    LIBRARY = "library"
    APPLICATION = "appl"


@click.command(name="context", help="Create vectors to classify contexts of annotatables")
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
    help="Path of project that was inferred over",
)
@click.option(
    "-t",
    "--tool",
    type=click.Choice(
        choices=SUPPORTED_TOOLS,
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: factory(value),
    required=True,
    help="Which inference tool was used Supported inference methods",
)
@click.option(
    "-rv",
    "--remove-var-annos",
    is_flag=True,
    help="Were variable annotations removed?",
)
@click.option(
    "-rp",
    "--remove-param-annos",
    is_flag=True,
    help="Were parameters annotations removed?",
)
@click.option(
    "-rr",
    "--remove-ret-annos",
    is_flag=True,
    help="Were return annotations removed?",
)
@click.option(
    "-l",
    "--loop",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a loop else 0",
)
@click.option(
    "-r",
    "--reassigned",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff variable is assigned to multiple times within same scope else 0",
)
@click.option(
    "-n",
    "--nested",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a nested scope else 0",
)
@click.option(
    "-f",
    "--flow",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff annotatable is in a branch (if-else) else 9",
)
@click.option(
    "-u",
    "--user-defined",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff given annotation is user-defined else 0",
)
def cli_entrypoint(
    inpath: pathlib.Path,
    tool: type[Inference],
    remove_var_annos: bool,
    remove_param_annos: bool,
    remove_ret_annos: bool,
    loop: bool,
    reassigned: bool,
    nested: bool,
    flow: bool,
    user_defined: bool,
) -> None:
    features = RelevantFeatures(
        loop=loop,
        reassigned=reassigned,
        nested=nested,
        builtin=user_defined,
        branching=flow,
    )

    removed = []

    if remove_var_annos:
        removed.append(TypeCollectionCategory.VARIABLE)
        removed.append(TypeCollectionCategory.VARIABLE)

    if remove_param_annos:
        removed.append(TypeCollectionCategory.CALLABLE_PARAMETER)

    if remove_ret_annos:
        removed.append(TypeCollectionCategory.CALLABLE_RETURN)

    inferred = output.read_inferred(inpath, tool.method(), removed=removed)

    for _, topx in inferred.groupby(by=InferredSchema.topn):
        with scratchpad(inpath) as sc:
            print(f"Using {sc} as a scratchpad for context gathering!")
            result = codemod.parallel_exec_transform_with_prettyprint(
                transform=TypeAnnotationApplierTransformer(
                    codemod.CodemodContext(), annotations=topx
                ),
                files=codemod.gather_files([str(sc)]),
                jobs=worker_count(),
                repo_root=str(sc),
            )
        print(format_parallel_exec_result(action="Annotation Application", result=result))

        df = generate_context_vectors_for_project(features, repo=sc)
        print(f"Feature set size: {df.shape}; writing to {output.context_vector_path(inpath)}")
        output.write_context_vectors(df, inpath)


if __name__ == "__main__":
    cli_entrypoint()
