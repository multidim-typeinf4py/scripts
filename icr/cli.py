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
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, path_type=pathlib.Path),
    required=False,
)
def entrypoint(
    static: list[type[Inference]],
    prob: list[type[Inference]],
    engine: type[ConflictResolution] | None,
    input: pathlib.Path,
    output: pathlib.Path | None,
) -> None:
    inf_count = len(static) + len(prob)
    assert inf_count != 0, "At least one inference method must be specified!"

    if inf_count > 1:
        assert (
            engine is not None
        ), "When specifiying multiple inference methods, an engine must be specified!"

    statics = list(map(lambda ctor: ctor(input), static))
    probabilistics = list(map(lambda ctor: ctor(input), prob))

    infs = lambda: itertools.chain(statics, probabilistics)

    # for inference in infs():
    #    inference.infer()

    # inferences = {inference.method: inference.inferred for inference in infs()}

    # for inferrer in infs():
    #     print(inferrer.inferred)

    if engine is not None:
        baseline = _collect(root=input)[1].df

        eng = engine(project=input, reference=baseline.drop(columns=["anno"], axis=1))
        inference = eng.resolve(
            probabilistics[0].inferred, dynamic=None, probabilistic=probabilistics[1].inferred
        )

        # print(inference)

    else:
        inference = next(infs()).inferred

    # if output is not None:


if __name__ == "__main__":
    entrypoint()
