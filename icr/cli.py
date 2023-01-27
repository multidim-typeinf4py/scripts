import pathlib
import itertools
import shutil
import sys

import click
from common import output
from common.schemas import InferredSchema, TypeCollectionSchema
from icr.insertion import TypeAnnotationApplierTransformer, HintRemover

from symbols.collector import build_type_collection
from .resolution import ConflictResolution, SubtypeVoting, Delegation
from .inference import Inference, MyPy, PyreInfer, PyreQuery, TypeWriter, Type4Py, HiTyper
from . import _factory

from libcst import codemod
from libcst.codemod import _cli as cstcli


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
    callback=lambda ctx, _, value: [_factory._inference_factory(v) for v in value] if value else [],
    required=False,
    multiple=True,
    help="Static inference methods",
)
@click.option(
    "-p",
    "--prob",
    type=click.Choice(
        choices=[HiTyper.__name__.lower(), TypeWriter.__name__.lower(), Type4Py.__name__.lower()],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: [_factory._inference_factory(v) for v in value] if value else [],
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
    callback=lambda ctx, _, value: _factory._engine_factory(value) if value else None,
    required=False,
    help="How differing inferences should be resolved",
)
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Project to intelligently infer over",
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
    "-r", "--remove-annos", is_flag=True, help="Remove all annotations in the codebase before inferring"
)
def entrypoint(
    static: list[type[Inference]],
    prob: list[type[Inference]],
    engine: type[ConflictResolution] | None,
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

    if remove_annos:
        inpath = inpath.parent / f"{inpath.name} - cleaned"
        if not inpath.is_dir():
            shutil.copytree(original, inpath)

            print(f"Removing annotations on '{inpath}'")

            codemod.parallel_exec_transform_with_prettyprint(
                transform=HintRemover(codemod.CodemodContext()),
                files=cstcli.gather_files([str(inpath)]),
                jobs=1,
                repo_root=str(inpath),
            )

    statics = list(map(lambda ctor: ctor(inpath), static))
    probabilistics = list(map(lambda ctor: ctor(inpath), prob))

    infs = lambda: itertools.chain(statics, probabilistics)

    for inference in infs():
        inference.infer()

    # inferences = {inference.method: inference.inferred for inference in infs()}

    # for inferrer in infs():
    #     print(inferrer.inferred)

    if engine is not None:
        baseline = build_type_collection(root=inpath).df

        eng = engine(
            project=inpath, reference=baseline.drop(columns=[TypeCollectionSchema.anno], axis=1)
        )
        inference_df = eng.resolve(
            probabilistics[0].inferred, dynamic=None, probabilistic=probabilistics[1].inferred
        )

    else:
        inference_df = next(infs()).inferred

        inference_df = inference_df.loc[
            inference_df.groupby(
                by=[InferredSchema.file, InferredSchema.category, InferredSchema.qname_ssa],
                sort=False,
            )[InferredSchema.topn].idxmin()
        ]

    if persist:
        outdir = _derive_output_folder(original, infs=list(infs()), engine=engine)
        if outdir.is_dir() and not overwrite:
            raise RuntimeError(
                f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
            )

        elif outdir.is_dir() and overwrite:
            shutil.rmtree(outdir)

        # outdir.mkdir(exist_ok=True, parents=True)
        shutil.copytree(inpath, outdir, symlinks=True)
        print("Applying annotations to code")

        # Retain top1 for annotating
        result = codemod.parallel_exec_transform_with_prettyprint(
            transform=TypeAnnotationApplierTransformer(
                context=codemod.CodemodContext(), tycol=inference_df
            ),
            files=cstcli.gather_files([str(outdir)]),
            jobs=1,
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

        output.write_icr(inference_df, outdir)
        print(f"Inferred types have been stored at {outdir}; exiting...")

    else:
        print(inference_df)
        print("Not persisting; exiting...")

    if remove_annos:
        shutil.rmtree(inpath)


def _derive_output_folder(
    inpath: pathlib.Path,
    /,
    infs: list[Inference],
    engine: type[ConflictResolution] | None = None,
) -> pathlib.Path:
    return (
        inpath.parent
        / f"{inpath.name}@{engine.method if engine else ''}({'+'.join(m.method for m in infs)})"
    )


if __name__ == "__main__":
    entrypoint()
