import itertools
import pathlib
import shutil
import sys
from typing import Optional

import click

from src import utils
from src.common import factory
from src.common import output
from src.common import TypeCollectionSchema
from src.infer.inference.hit4py import HiTyper

from src.icr.resolution import ConflictResolution
from src.infer.inference import MyPy, PyreInfer, PyreQuery, Type4Py, _TypeWriter

from libcst import codemod

from src.icr.resolution import SubtypeVoting, Delegation
from src.infer.insertion import TypeAnnotationApplierTransformer
from src.symbols.collector import build_type_collection


@click.command(
    name="icr",
    help="Intelligent Conflict Resolution on multiple inference methodologies",
)
@click.option(
    "-s",
    "--static",
    type=click.Choice(
        choices=[MyPy.__name__.lower(), PyreInfer.__name__.lower(), PyreQuery.__name__.lower()],
        case_sensitive=False,
    ),
    required=False,
    multiple=True,
    help="Static inference methods",
)
@click.option(
    "-p",
    "--prob",
    type=click.Choice(
        choices=[HiTyper.__name__.lower(), _TypeWriter.__name__.lower(), Type4Py.__name__.lower()],
        case_sensitive=False,
    ),
    required=False,
    multiple=True,
    help="Probabilistic inference methods",
)
@click.option(
    "-e",
    "--engine",
    type=click.Choice(
        choices=[
            SubtypeVoting.__name__.lower(),
            Delegation.__name__.lower(),
        ],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: factory._engine_factory(value) if value else None,
    required=False,
    help="How differing inferences should be resolved",
)
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="(Original) Project to intelligently infer over",
)
@click.option(
    "-o",
    "--persist",
    is_flag=True,
    default=False,
    help="Indicate whether the resolved hints should be stored",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the target folder. Has no effect if --output was not given",
)
@click.option(
    "-r",
    "--remove-annos",
    is_flag=True,
    help="Remove all annotations in the codebase before inferring",
)
def cli_entrypoint(
    static: list[str],
    prob: list[str],
    engine: Optional[type[ConflictResolution]],
    inpath: pathlib.Path,
    persist: bool,
    overwrite: bool,
    remove_annos: bool,
) -> None:
    original = inpath
    inf_count = len(static) + len(prob)
    assert inf_count != 0, "At least one inference method must be specified!"

    if inf_count > 1:
        assert (
            engine is not None
        ), "When specifiying multiple inference methods, an engine must be specified!"

    tool2outputdir = {
        tool: output.inference_output_path(inpath, tool) for tool in itertools.chain(static, prob)
    }

    tool2icr = {
        tool: output.read_inferred(output_dir) for tool, output_dir in tool2outputdir.items()
    }

    if engine is not None:
        baseline = build_type_collection(root=inpath).df

        eng = engine(
            project=inpath, reference=baseline.drop(columns=[TypeCollectionSchema.anno], axis=1)
        )

        tool = "+".join(tool2icr.keys())
        inference_df = eng.resolve(tool2icr)

    else:
        tool, icr_df = next((tool, df) for tool, df in tool2icr.items())

        inference_df = utils.top_preds_only(icr_df)

    if persist:
        outdir = output.inference_output_path(original, tool=tool)
        if outdir.is_dir() and not overwrite:
            raise RuntimeError(
                f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
            )

        elif outdir.is_dir() and overwrite:
            shutil.rmtree(outdir)

        # outdir.mkdir(exist_ok=True, parents=True)
        shutil.copytree(inpath, outdir, ignore_dangling_symlinks=True, symlinks=True)

        output.write_inferred(inference_df, outdir)
        print(f"Inferred types have been stored at {outdir}")

        print("Applying annotations to code")
        # Retain top1 for annotating
        result = codemod.parallel_exec_transform_with_prettyprint(
            transform=TypeAnnotationApplierTransformer(
                context=codemod.CodemodContext(), annotations=inference_df
            ),
            jobs=utils.worker_count(),
            files=codemod.gather_files([str(outdir)]),
            repo_root=str(outdir),
        )
        print(
            f"Finished codemodding {result.successes + result.skips + result.failures} files!",
            file=sys.stderr,
        )
        print(
            f" - Collected symbol from {result.successes} files successfully.",
            file=sys.stderr,
        )
        print(f" - Skipped {result.skips} files.", file=sys.stderr)
        print(f" - Failed to collect from {result.failures} files.", file=sys.stderr)
        print(f" - {result.warnings} warnings were generated.", file=sys.stderr)

    else:
        print(inference_df)
        print("Not persisting; exiting...")

    if remove_annos:
        shutil.rmtree(inpath)


if __name__ == "__main__":
    cli_entrypoint()
