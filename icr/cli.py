import pathlib
import itertools

import click

from symbols.cli import _collect
from .resolution import ConflictResolution, SubtypeVoting, Delegation
from .inference import Inference, MyPy, PyreInfer, TypeWriter, Type4Py, HiTyper
from . import _factory


@click.command(
    name="icr",
    help="Intelligent Conflict Resolution on multiple inference methodologies",
)
@click.option(
    "-s",
    "--static",
    type=click.Choice(
        choices=[MyPy.__name__.lower(), PyreInfer.__name__.lower()], case_sensitive=False
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
def entrypoint(
    static: list[type[Inference]],
    prob: list[type[Inference]],
    engine: type[ConflictResolution] | None,
    inpath: pathlib.Path,
    persist: bool,
    overwrite: bool,
) -> None:
    inf_count = len(static) + len(prob)
    assert inf_count != 0, "At least one inference method must be specified!"

    if inf_count > 1:
        assert (
            engine is not None
        ), "When specifiying multiple inference methods, an engine must be specified!"

    statics = list(map(lambda ctor: ctor(inpath), static))
    probabilistics = list(map(lambda ctor: ctor(inpath), prob))

    infs = lambda: itertools.chain(statics, probabilistics)

    for inference in infs():
       inference.infer()

    # inferences = {inference.method: inference.inferred for inference in infs()}

    # for inferrer in infs():
    #     print(inferrer.inferred)

    if engine is not None:
        baseline = _collect(root=inpath)[1].df

        eng = engine(project=inpath, reference=baseline.drop(columns=["anno"], axis=1))
        inference = eng.resolve(
            probabilistics[0].inferred, dynamic=None, probabilistic=probabilistics[1].inferred
        )

    else:
        inference = next(infs()).inferred

    if persist:
        outdir = _derive_output_folder(inpath, infs=list(infs()), engine=engine)
        if outdir.is_dir() and not overwrite:
            raise RuntimeError(
                f"--overwrite was not given! Refraining from deleting already existing {outdir=}"
            )

        outdir.mkdir(exist_ok=True, parents=True)
        outpath = outdir / ".icr.csv"
        inference.to_csv(outpath)
        print(f"Inferred types have been stored at {outpath}; exiting...")

    else:
        print("Not persisting; exiting...")


def _derive_output_folder(
    inpath: pathlib.Path,
    /,
    infs: list[Inference],
    engine: type[ConflictResolution] | None = None,
) -> pathlib.Path:
    return inpath.parent / f"{inpath.name} - {engine.method if engine else ''}({'+'.join(m.method for m in infs)})"


if __name__ == "__main__":
    entrypoint()
