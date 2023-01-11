import enum
import pathlib
import sys
import click
import pandera.typing as pt

import libcst.codemod as codemod


from context.features import RelevantFeatures

from common.schemas import ContextSymbolSchema, ContextSymbolSchemaColumns
from .visitors import ContextVectorMaker


class Purpose(str, enum.Enum):
    LIBRARY = "library"
    APPLICATION = "appl"


@click.command(name="context", help="Create vectors to classify contexts of annotatables")
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
    help="Project to analyse",
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
    "-u",
    "--user-defined",
    is_flag=True,
    show_default=True,
    default=False,
    help="1 iff given annotation is user-defined else 0",
)
#@click.option(
#    "-s",
#    "--scope",
#    is_flag=True,
#    show_default=True,
#    default=False,
#    help="Generate numeric indicator for annotatable's scope; GlobalScope, FunctionScope, etc.",
#)
# @click.option(
#     "-p",
#     "--purpose",
#     type=click.Choice(["library", "application"]),
#     callback=lambda ctx, _, val: Purpose[val] if val is not None else None,
#     help="Supply library purpose",
# )
def entrypoint(
    inpath: pathlib.Path,
    loop: bool,
    reassigned: bool,
    nested: bool,
    user_defined: bool,
    #scope: bool,  # , purpose: Purpose
) -> None:
    vmaker = ContextVectorMaker(
        context=codemod.CodemodContext(),
        features=RelevantFeatures(
            loop=loop, reassigned=reassigned, nested=nested, user_defined=user_defined# , scope=scope
        ),
    )

    presult = codemod.parallel_exec_transform_with_prettyprint(
        transform=vmaker,
        files=codemod.gather_files(
            [str(inpath)],
        ),
        jobs=1,
        repo_root=str(inpath),
    )

    print(
        f"Finished codemodding {presult.successes + presult.skips + presult.failures} files!",
        file=sys.stderr,
    )
    print(
        f" - Collected symbol from {presult.successes} files successfully.",
        file=sys.stderr,
    )
    print(f" - Skipped {presult.skips} files.", file=sys.stderr)
    print(f" - Failed to collect from {presult.failures} files.", file=sys.stderr)
    print(f" - {presult.warnings} warnings were generated.", file=sys.stderr)

    df = pt.DataFrame[ContextSymbolSchema](vmaker.dfrs, columns=ContextSymbolSchemaColumns)
    print(df)


if __name__ == "__main__":
    entrypoint()
